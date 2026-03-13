ultrathink 现在我需要继续完成消融试验和端到端实验，但是由于很久没改了，面对代码有点难以适应，需要更多的交给claude完成后续代码：
  - 为了帮助未来claude开发feature的时候快速理解每个部分的代码，可以递归探索codebase
  - 从当前消融试验脚本出发，利用superpower，给出探索方案。
  - 使用 feature-dev插件引导探索
  - 为了避免上下文溢出，尽量切割为相对独立的探索任务，建立agent-team驱动多个sub-agent 并行探索，但是要注意并行度<5。
  - 如有必要，在doc文件夹下生成，相应的文档，并建立层次化索引，最高层为claude.md。使用插件 spec-writer
  - 注意仓库有很多废弃代码，如果你发现了废弃代码，可以用专门文档记录一下。
  - 项目的论文思路见 @doc/spec/key_COT.py
  - 技术思路见：scheduler_paper/main.tex


在 improve claude md 之后
Ultrathink, feature-dev, superpower:
  启动一个独立agent，
  检查 CLAUDE.md 删除的内容（上面的修改历史我存在了 @doc/dev/claude_revise.md  ），是否可以在文档索引的index
  tree中找到？是否一致？另外，由于当前正在调试的是消融实验，消融实验的细节位置，需要在claude.md中突出记录。