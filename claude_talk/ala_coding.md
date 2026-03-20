ultrathink 现在我需要继续完成消融试验和端到端实验，但是由于很久没改了，面对代码有点难以适应，需要更多的交给claude完成后续代码：
  - 为了帮助未来claude开发feature的时候快速理解每个部分的代码，可以递归探索codebase
  - 从当前消融试验脚本出发，利用superpower，给出探索方案。
  - 使用 feature-dev插件引导探索
  - 为了避免上下文溢出，尽量切割为相对独立的探索任务，建立agent-team驱动多个sub-agent 并行探索，但是要注意并行度<5。
  - 如有必要，在doc文件夹下生成，相应的文档，并建立层次化索引，最高层为claude.md。使用插件 spec-writer
  - 注意仓库有很多废弃代码，如果你发现了废弃代码，可以用专门文档记录一下。
  - 项目的论文思路见 @doc/spec/key_COT.py
  - 技术思路见：scheduler_paper/main.tex

Ultrathink, feature-dev, superpower:
  启动一个独立agent，
  检查 CLAUDE.md 删除的内容（上面的修改历史我存在了 @doc/dev/claude_revise.md  ），是否可以在文档索引的index
  tree中找到？是否一致？另外，由于当前正在调试的是消融实验，消融实验的细节位置，需要在claude.md中突出记录。



请 Ultrathink，调用superpower, sequential thinking MCP 工具，解决以下问题:
根据 @claude_talk/auto-abla.md，依次开始，[理解流程步骤]，[生成自动化流程]，[开始自动化调试]，完成，计划，端到端调试的流程，直到完全成功。


[空间分析]
自变量维度：方法，负载（三个参数组合），方法参数（分箱 或者ration B）
因变量维度：切换开销，切换次数，延迟分解，miss rate

图的维度：
  - 自变量展示：簇，簇内X
  - 因变量展示：y轴
  - 公共展示：小节，图，子图

方案设计本质上，是将变量维度映射到图的维度。

[敲定方案]

负载选定三种配置：9 chains
  低 400T-0.5×
  中 400T-1.0×
  高 200T-1.0×

小节1 消融试验2：
  扫描分箱

  图一:
  - Y1: realloc count 柱子 Y2：realloc ratio 折现
  - 簇：负载，簇内X：bin

  图二：
  - Y1: latency breakdown Y2：miss rate
  - 簇：负载，簇内X：bin

小节2 消融试验3：
  扫描ratio B，选定分箱=8

  图一:
  - Y1: realloc count 柱子 Y2：realloc ratio 折现
  - 簇：负载，簇内X：ratioB

  图二：
  - Y1: latency breakdown Y2：miss rate
  - 簇：负载，簇内X：ratioB


当前的方案，已经初具雏形，特别是 消融试验2, 只需要换一下x 轴的分组方式，减少一下绘制的负载配置就行
- 修改绘图函数，注意不要篡改了，motiv实验的函数，如果改了需要改回来
- 不要产生没用的图


Ultrathink:
- 使用superpower 规划修改和测试
- 添加测试用的代码，保证输出辅助信息帮助确认repack的执行情况和成功失败情况统计。
- 启动一个subagent 测试，可以使用python-debug 技能
- 确定repack 真有有开始在执行，而不是一直卡着
- 现在的cyc-S 和 reserv 都能触发repack了 确认现在repack的执行情况。