简单总结一下 run.py 和 Codex app-server   之间的通信内容有哪些，比如 run 向 Codex app-server 会有 request 'turn/start' 等:
`run.py` 和 Codex app-server 之间主要通过以下消息通信。

## `run.py` → Codex app-server

### RPC Request：需要等待返回结果

- `initialize`：初始化 JSON-RPC 会话，声明客户端信息和能力。
- `thread/start`：创建贯穿整个 episode 的持久化 Codex thread。
- `turn/start`：在已有 thread 中启动一次模型运行。
- `thread/read`：网络异常时，只读查询 thread 和 turn 是否已经结束。

### RPC Notification：不等待返回

- `initialized`：通知 app-server 初始化握手已经完成。

### RPC Reply：回复 app-server 的工具请求

- 回复 `item/tool/call`：将 `robodojo_start`、`pi05_infer`、`robodojo_execute` 或 `robodojo_act` 的执行结果返回给 Codex。

---

## Codex app-server → `run.py`

### RPC Response

对应上述 request 返回结果，例如：

- `initialize` 的初始化结果；
- `thread/start` 返回 `thread_id`；
- `turn/start` 返回 `turn_id`；
- `thread/read` 返回 thread 当前状态。

### 主动发送的事件或请求

- `item/tool/call`：模型请求调用一个 RoboDojo 动态工具。
- `item/started`：Codex 开始执行原生工具或生成内容。
- `item/completed`：Codex 完成原生工具或公开消息。
- `thread/tokenUsage/updated`：更新当前 thread 的 token 用量。
- `turn/completed`：本次模型 turn 已结束。
- `turn/failed`：本次 turn 失败。
- `error`：app-server 或模型连接发生错误。

整体可以简化为：

```text
run.py                              Codex app-server
  │                                       │
  │── request: initialize ───────────────>│
  │<──────────── response ────────────────│
  │── notify: initialized ───────────────>│
  │── request: thread/start ─────────────>│
  │<──── thread_id ───────────────────────│
  │── request: turn/start ───────────────>│
  │<──── turn_id ─────────────────────────│
  │                                       │
  │<── event: item/tool/call ─────────────│
  │── reply: 工具执行结果 ───────────────>│
  │                                       │
  │<── event: item/started/completed ─────│
  │<── event: tokenUsage/updated ─────────│
  │<── event: turn/completed ─────────────│
```

一句话概括：

> `run.py` 负责创建和驱动 Codex thread/turn，并执行模型发出的 RoboDojo 工具调用；Codex app-server 负责运行模型，并持续向 `run.py` 推送工具请求、执行事件、token 用量和 turn 终态。

---

在 run 中，相关的调用函数是哪些？比如 RPC Request   就是 self.transport.request
但是Codex app-server → run.py   的 RPC Response   和  主动发送的事件或请求   呢？:

对，`run.py` 主动发送 RPC Request 使用：

```python
self.transport.request(...)
```

而 Codex app-server 返回的消息分为两类，它们走不同的接收路径。

# 1. RPC Response：由 `request()` 内部接收

例如 `run.py` 调用：

```python
response = self.transport.request(
    'thread/start',
    params,
    timeout,
)
```

表面上只看到一次 `request()`，实际上它内部完成了“发送请求 + 等待响应”两个步骤。

相关函数调用链是：

```text
run.py
  self.transport.request(...)
        │
        ├── StdioAppServer._send()
        │     把 JSON-RPC 请求写入 app-server stdin
        │
        └── destination.get()
              阻塞等待响应
```

同时，后台线程运行：

```text
StdioAppServer._read_stdout()
    │
    ├── 从 app-server stdout 读取一行 JSON
    ├── 根据响应中的 id 找到 destination 队列
    └── destination.put(message)
```

`request()` 被唤醒后返回：

```python
return response.get("result")
```

因此，RPC Response 在 `run.py` 中没有单独的 `receive_response()` 调用。

它被封装在：

```python
self.transport.request(...)
```

内部。

例如：

```python
response = self.transport.request('thread/start', ...)
self.thread_id = response['thread']['id']
```

这里的 `response` 就是 app-server 返回的 RPC Response 中的 `result`。

---

# 2. app-server 主动发送的事件：`next_message()`

Codex app-server 主动发送的事件不是某个 `request()` 的直接响应，例如：

```text
item/started
item/completed
thread/tokenUsage/updated
turn/completed
turn/failed
error
```

这些消息由后台线程 `_read_stdout()` 放入统一事件队列：

```python
self._incoming.put(message)
```

然后 `run.py` 的事件循环通过：

```python
event = self.transport.next_message(remaining)
```

取出一条消息。

相关调用链是：

```text
Codex app-server stdout
        │
        ▼
StdioAppServer._read_stdout()
        │
        ▼
self._incoming.put(message)
        │
        ▼
CodexPolicy.run()
        │
        ▼
self.transport.next_message(remaining)
```

`run.py` 随后读取：

```python
method = event.get('method')
params = event.get('params', {})
```

再根据 `method` 分支处理：

```python
if method == 'thread/tokenUsage/updated':
    ...

elif method == 'item/tool/call':
    ...

elif method in ('item/started', 'item/completed'):
    ...

elif method == 'turn/completed':
    ...

elif method in ('error', 'turn/failed'):
    ...
```

---

# 3. app-server 主动发来的工具请求

`item/tool/call` 比普通事件特殊，因为它是 app-server 对 `run.py` 发起的反向 RPC Request。

接收仍然使用：

```python
event = self.transport.next_message(remaining)
```

识别后：

```python
elif method == 'item/tool/call':
```

`run.py` 找到对应 handler：

```python
handlers = {
    'robodojo_start': rollout.start,
    'pi05_infer': rollout.infer,
    'robodojo_execute': rollout.execute,
}
```

然后调用：

```python
packet = handlers[name](**arguments)
```

最后回复 app-server：

```python
self.transport.reply(
    event['id'],
    {
        'success': success,
        'contentItems': ...,
    },
)
```

完整路径是：

```text
Codex app-server
    │ 发送 item/tool/call，包含 id
    ▼
StdioAppServer._read_stdout()
    │
    ▼
self._incoming
    │
    ▼
self.transport.next_message()
    │
    ▼
run.py 调用 rollout handler
    │
    ▼
self.transport.reply(event['id'], result)
    │
    ▼
Codex app-server
```

---

# 4. 三种情况对应的函数

可以简单记成：

```text
run.py 主动请求 app-server：
    self.transport.request(...)

run.py 单向通知 app-server：
    self.transport.notify(...)

run.py 等待 app-server 主动消息：
    self.transport.next_message(...)

run.py 回复 app-server 的反向请求：
    self.transport.reply(...)
```

底层辅助函数则是：

```text
发送所有消息：
    StdioAppServer._send()

后台读取 stdout：
    StdioAppServer._read_stdout()

后台读取 stderr：
    StdioAppServer._read_stderr()
```

# 5. `_read_stdout()` 如何区分两类消息

核心判断是：

```python
identifier = message.get("id")

if identifier is not None and "method" not in message:
    # 这是 run.py 之前发出 request 的响应
    destination.put(message)
else:
    # 这是事件、通知，或者 app-server 发来的反向请求
    self._incoming.put(message)
```

也就是：

### 有 `id`，没有 `method`

```json
{
  "id": 1,
  "result": {}
}
```

这是 RPC Response，交给正在等待的 `request()`。

### 有 `method`

```json
{
  "method": "turn/completed",
  "params": {}
}
```

这是主动事件，交给 `next_message()`。

### 同时有 `id` 和 `method`

```json
{
  "id": 37,
  "method": "item/tool/call",
  "params": {}
}
```

这是 app-server 发给 `run.py` 的反向 RPC Request：

- 使用 `next_message()` 接收；
- 使用 `reply(event['id'], ...)` 回复。

因此最核心的对应关系是：

```text
request()      <──内部接收── RPC Response

next_message() <──接收──── app-server 主动事件或反向 Request

reply()        ──回复────── app-server 的反向 Request
```