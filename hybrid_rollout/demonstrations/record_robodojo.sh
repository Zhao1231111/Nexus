#!/usr/bin/env bash
# 独立的演示录制入口。无需 OpenPI、Codex 账号或策略服务。
set -euo pipefail

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_SOURCE="${ROBODOJO_SOURCE:?Set the external RoboDojo source directory}"
ROBODOJO_ENV="${ROBODOJO_ENV:?Set the verified Isaac Sim 6.0.1 environment prefix}"
DEMO_LOG="${DEMO_LOG:?Set a fresh recording log file path}"
[[ ! -e "$DEMO_LOG" ]] || { echo "日志已存在，请选择新路径: $DEMO_LOG" >&2; exit 2; }
mkdir -p -- "$(dirname -- "$DEMO_LOG")"
export PYTHONPATH="$CODE_ROOT:$ROBODOJO_SOURCE:$ROBODOJO_SOURCE/XPolicyLab:$ROBODOJO_SOURCE/third_party/curobo${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
# 与现有 RoboDojo 运行时一致，优先使用环境自身的 C++ 运行库。
export LD_LIBRARY_PATH="$ROBODOJO_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$ROBODOJO_ENV/bin/python" -c 'import importlib.metadata as m; assert m.version("isaacsim").startswith("6.0.1")'
cd -- "$ROBODOJO_SOURCE"
# 在有界冒烟中，停止 conda 包装进程后曾遗留 Isaac 子进程。这里为当前录制
# 创建独立进程组，退出时只清理该组，不影响用户已有的模拟器或其他任务。
demo_pid=''
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  if [[ -n "$demo_pid" ]] && kill -0 -- "-$demo_pid" 2>/dev/null; then
    kill -TERM -- "-$demo_pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 -- "-$demo_pid" 2>/dev/null || break
      sleep .5
    done
    kill -KILL -- "-$demo_pid" 2>/dev/null || true
  fi
  exit "$exit_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set +e
setsid "$ROBODOJO_ENV/bin/python" -m hybrid_rollout.demonstrations.robodojo "$@" >"$DEMO_LOG" 2>&1 &
demo_pid=$!
wait "$demo_pid"
status=$?
set -e
# Kit 关闭时有时会覆盖 Python 退出码，因此同时要求明确的录制完成标记。
if (( status != 0 )) || ! rg -q '^DEMO_PASS ' "$DEMO_LOG"; then
  echo "演示录制失败 (status=$status)，日志: $DEMO_LOG" >&2
  exit 1
fi
# 原日志保留回车进度条，终端摘要将回车转换为换行，避免长进度条粘在阶段标记上。
tr '\r' '\n' < "$DEMO_LOG" | rg '^DEMO_(STAGE|PROGRESS|RESET_PROGRESS|PASS)'
