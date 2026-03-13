[理解流程步骤]
请 Ultrathink，调用superpower, sequential thinking MCP 工具，理解以下流程:
端到端调试，完整流程有四步：

1. code review，待检查项如下：
  - 确认各个case （cyc,glb,pglb,reserv,cyc-S）触发的执行流程是否符合 @doc/spec，下的test_plan.md **执行路径对照** 部分的
  确认 各个case 触发的执行路径，没有违反 @doc/spec/readme.md ## 4. 关键澄清与约定
  - 预告：
    - 之前motiv是验证过，cyc和 glb的
    - 剩下的三种没有review过
  - 由于模块化比较好，（cyc,glb,pglb,reserv,cyc-S）对应执行路径有两类：
    - 调用层面的行为
    - 函数本身的行为
  - 已知问题：
    - 已知，cyc-S 在消融试验脚本中 通过runtime={'policy': 'reserv'} 指定了 运行时算法，但是这种调用会不会出现参数传递的问题，仍需要确认，即，是否将reserv需要的参数（PartitionConfig）都传过去了。
    - runtime={'policy': 'cyc-S'} 本身用到的运行时算法却是另一种policy，在现在的case 下似乎用不到。
    - 如果调用层面能保证，调度算法所需要的数据结构正确传递那么就没问题。
2. 测试：
  确认启动 conda的 gurobi环境，确认执行正确的命令。
    - 不报错
    - 能执行完
    - 如果出错了
      - 可以使用 /python-debuger 插件
      - 如果报错被 python 多线程和 try--expectation 逻辑拦截了，可以切换到单线程，以及注释掉try逻辑
      - 可以用简单的例子测试，知道报错消除
3. 结果检查
  * 检查数据，是否正常
  * 检查绘图函数和图例，是否和 key_COT 需求对齐
  * 检查图片，可以使用 图片理解 mcp (zai-mcp-server:analyze_image,analyze_data_visualization)
    * 已知问题：三个case 都生成了很多图，但是意义不明
4. 反复调教图片格式（参考 论文 scheduler_paper.tex

[生成自动化流程]
  请使用  /claude-automation-recommender 和 superpower 规划执行流程

  例如：线性执行，出错了就回滚重头来。
  流程投入使用前，需确认复合以下特性：
    流程需保证不能因为超上下文导致，流程崩掉：
        - 即使落盘记忆
        - 善用 agent-team
        - 善主动压缩记忆
    代码修改使能：
        ultrathink 和 feature-dev，superpower插件
    如果提示 请求达到使用上限，需要使用 cc-switch-provider 切换，并选择 model opus

[开始自动化调试]
  开启agent team
  /ralph-loop 直到成功

装箱算法被 bypass — bin_list 保留 Phase 1 的空间布局；TODO 未来为 reserv 启用
完成所有步骤后，调试装箱算法

后续repack算法:
完成了无装箱流程之后，可以调试装箱算法
1. 如果真是因为ratioB 变得保守导致资源分配不过来，repack失败是正常的，但是，我感觉，如果ratioB <= ratio A
那么失败是不正常的，特别，是cyc-S的repack，因为所有的item都变小了，不应该会失败。一定是某种参数传递错误，导致repack阶段，箱子变小或者item变大了

2. repack
失败的处理：repack失败之前选择报错，然后退出，这样是可以接受的。如果不报错的话，会出现，repack失败了最终的箱子不完整覆盖所有任务。可以选择：
推荐方案 A：在 extract_pid2_bin_id 清空 bin 之前备份 scheduling 信息，repack 后将未成功放置的任务用 Phase 1
的原始信息恢复。这样保证所有任务始终在 partition 中。你认是否应该报错？
