# B22：资源依赖与实验公共 API 边界

状态：CLOSED / ACCEPTED（真实reviewer E0088），REQ-020。裁定KEEP_AS_IS；本批只有分析，没有源码清理授权。下文提案保留，末尾记录裁定与后续只读观察。

## 已核实的保留范围

沿sim_main读取资源模型、任务队列、monitor和scheduler入口、绘图、性能、消息上下文、屏障、LRU和位置表，再返回motiv/abla→exp_common支路。完整读取11个模块，旧Scheduler类和TaskInt只核相关上下文，不声称已审完类内实现。

- DDL_reservation/RT_reservation仍在TaskInt.update_sched_size中构造；dummy_reservation还有导入，保留整组，不清理类内方法。
- Monitor两个助手有预分配和旧runtime调用；performance的四个助手都有调用；bin_list_utils三函数用于实际绘图和结果输出。
- trace_analyser被analyze_timing调用，不能因不在三入口常规路径而归档。ContextMsg示例、Barrier测试、TaskInt诊断入口和旧runtime保持。

## 需要 reviewer 裁定的边界

scripts/exp_common.py五个助手：group_by_key、compute_group_means、setup_dual_axis_plot、save_and_close_figure、add_value_labels。共51行，171明确代码及tracked文本查询未发现仓内可执行引用；四个实际调用方只导入参数、运行入口、字体/标签等其他组件，动态调用点未指向五个助手。

但受保护的doc/guide/exp_scripts_overview.md:69-73仍把它们列为公共API，collector_overview.md:263-264也列出两个。2026-02-12的ABLA_EXP_FIX_PLAN.md还保留旧导入例子。这不是“零文档引用”，也不能假设所有仓外调用都不存在。

**提议KEEP_AS_IS。** 不退休原导出或pickle路径；不移动代码后留下失真的指南；不为缩短一个文件添加转发层。未来确需重新划分公开API或更新保护指南，应另提具体方案，不能套用本次只读审计作为授权。

## 验证与覆盖

当前范围171代码文件：155Python、16shell。155个AST可解析，三入口保守静态可达67文件；图包含main、TYPE_CHECKING和函数内导入，不等于实际运行可达性。原137Python扫描快照保留，当前进度只补latest_semantic_review字段。

全部171代码SHA与B21验收后一致。没有新测试运行，B21的259=254passed+原5失败仅作历史背景。HEAD/index未变，ledger无新增行，保护文档、测试和依赖未改。

## 后续与恢复

请真实reviewer独立核对保留依据并裁定五个助手是否KEEP_AS_IS。裁定前不进行源码动作，全仓语义审查仍未完成。下一步继续实验工作流剩余依赖，再沿已记录的孤立模块DFS顺序检查。

本批只改审计报告、协议和双全局记录，没有源码需要恢复。分析有误时追加更正，不能checkout整份全局记录或删除后续协议事件。没有提交推送授权。

## 独立裁定

E0088确认五个助手保持原样：两份保护指南仍公开其API，零仓内调用不能自动撤销契约。未来如需处置，要另提方案并由用户先裁定指南API条目的调整。资源/消息侧保留依据得到复核；Codex在裁定后全量核对171代码SHA，全部不变。

等待期间另读E2E入口：两个--help均rc0，日志/tmp/scheduler-e2e-entry-audit-g_vk883k，未运行完整实验或pytest。两个文件保留为独立CLI；round_to_step仅发现定义及历史候选行，目前只是REVIEW线索，不在E0088源码批准范围内，也未修改。完整证据位于JSON的post_review_followup。

本批没有源码需恢复。B11-B21十一源码批次仍未提交，B22不计入源码批次数。全仓语义审查未完成。
