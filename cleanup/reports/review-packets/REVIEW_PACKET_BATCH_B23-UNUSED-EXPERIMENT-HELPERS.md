# B23 实验助手迁移审查

状态：CLOSED / ACCEPTED，REQ-021，真实reviewer E0092已验收。以下提案范围、计划和执行阶段结果保留为历史；当前结论及补核见文末。未提交或推送。

## 待审范围

| 原位置 | 原样移入 | 行数 |
| --- | --- | --- |
| scripts/e2e_exp_runner.py::round_to_step | scripts/e2e_exp_runner_unused.py | 3 |
| analyze/stat_num_exec.py::extract_num_cores | analyze/stat_num_exec_unused.py | 7 |

只移动完整def块。原imports、空行、其他函数、Runner类、main与旧分析链保持原字节；第二份归档只另加import re。新增test_unused_experiment_helpers.py，不修改既有测试。
源剩余预计算SHA：
- e2e_exp_runner：e1dc1dd1f5803137d12e942df4169e69cb269f0e4a0651f50379bec96a1c7445
- stat_num_exec：0f25b6b6275d0473cedfb669a873b92f81c58ac736d4065aa1225a75fcd405ef

## 依据与风险

沿实验worker和历史分析依赖读了50个顶层函数。171明确代码及307可读tracked文件内，两候选只有自身定义和旧候选记录，未发现调用或保护指南API条目。stat_num_exec的两处实际导入只取extract_num_exec。
未跟踪内容不纳入审计，仓外使用未知。旧导出和对应函数pickle路径将退休，不加兼容层；公共名称集合精确41→40、29→28，各只减少目标一名。这是MEDIUM兼容风险，请独立reviewer明确裁定；不同意则保留。

## 已验证与保留

21原行为样本在两个独立gurobi Python进程一致，包括原异常、负数/非整数舍入、Unicode数字及首段数字匹配。9个独立CLI的help通过。
旧analyze_tp实际运行在131行报 AttributeError: 'Namespace' object has no attribute 'sim_seq'，迁移前已复现；不修复、不以help通过掩盖该失败。
run/1/all.sh、run/2/all2.sh仍调用分析链；UE_extract为独立CLI并被历史文档引用；test_mode为手工诊断入口。它们及全部类内部保留。E0088五个exp_common文档API继续KEEP_AS_IS。

## 实施和验收

1. 真实APPROVED后才新增测试，先跑目标存在性测试，以预期断言失败证明RED。
2. 原样移10行；块连续且各一次，源剩余SHA与EOF不变。stat_num_exec原无尾LF，e2e原有LF；工具误增减LF时不得改预期，必须在reviewer授权和完整字节断言下纠正。
3. 计划31项新测试：21行为、两源剩余、两归档原块、两接口集合/签名、两pickle边界、目标存在性及借用extract_num_exec identity。
4. 与既有13模块合跑，259=254+同5个collector失败为对照，最终以实际XML为准。不跳过、不修原失败。复跑main及9脚本help、旧吞吐已知失败。
5. 非本批169代码哈希、保护文档、依赖、HEAD/index不变；move追加2行14列，reachable追加3行10列，保留旧字节及历史坏行。
6. 持久化仅五代码文件的恢复patch并执行reverse --check，不实际恢复。同步双全局记录及报告；RESULT后等独立ACCEPTED。

详情：cleanup/reports/b23-unused-experiment-helpers.json。
日志：/tmp/scheduler-b23-preflight-6tni7hjm。
无commit/push授权；全仓语义审查未完成。

## 实际结果

新31/31通过；完整十四模块290=285passed+原5同名同异常collector失败，0error/skip。两源0/3、0/7，全字节与预计算相等；stat工具补LF已按E0090 N5还原。169非本批与当前174代码SHA复核，保护区/依赖/HEAD/index不变。main+9help通过，旧analyze_tp缺sim_seq失败保持；全部日志/tmp/scheduler-b23-validation-p05iy7gt。

账本前缀24806/73533B保留，新增2×14和3×10；8历史坏行保留。五路径恢复patch已持久化JSON，reverse--check0未恢复。完整接口和pickle边界通过。XLSL额外跟进只读：5模块help过，3条旧直接入口缺analyze包，不修。E0091为当时的待验收结果，现由E0092关闭，不自动提交。

## 独立验收与收尾

真实reviewer E0092接受B23。Codex直接解析reviewer的/tmp/b23_review_final/b23only.xml与fourteen.xml：31全过；十四模块290=285通过+原5同名同异常collector失败，0error/skip，模块计数逐一相同。

同一reviewer会话补做169非本批SHA逐键核对，并按原11条validation.cli命令复跑：10个help全0，旧analyze_tp仍以1退出并报sim_seq AttributeError。CLI补核见/tmp/b23_review_final/n3-supplement.json；repack_sweep不是本批范围。Codex另重算全部169非本批及174当前代码SHA，均不变；两源全字节、EOF、账本前缀及限定恢复patch检查通过。原diff-check空行警告仍保留。

只读后续：debug_sink_constraint保留；example/bm1、bm2、bm3仍REVIEW保留；mapper/mem_planner仅头部与AST初查，三项线索不构成迁移授权。全仓审查未完成。当前状态和历史分别见CLEANUP_STATUS.md、FILE_ADJUSTMENT_RECORD.md；B11–B21加B23共十二源码批次未提交，B22只分析。
