# [AGENTS.md](http://AGENTS.md)

## Project scope

## Coding rules

代码必须方便 review。给代码加上详细的中文注释，尤其是行内注释。

## Experiments and launch scripts



## Running Experiments by sub-agnet luna median

对于输出较长的程序（如gaussian-lic），使用 sub-agent luna 运行实验，并负责分析日志，在首次异常或完成时向主agent汇报，从而避免污染主agent的上下文；
sub-agent 已经负责日志分析后，主agent不应再重复读取较长的日志，主agent可做少量合理的状态检查从而确认实验没有挂住；
调用 sub-agent 时，主agent需要根据任务复杂度自行判断 fork_turns 的具体取值：我们希望尽可能节约消耗额度，但是也要保证 sub-agent 的理解、分析效果。

## Model and dataset contracts



## Verification and third-party code



## Documentation and reporting

