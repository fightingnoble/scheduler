# B19：同一依赖路径上的未用助手

状态：ACCEPTED，REQ-017，真实reviewer E0076及其补核。已实施、已独立验收，尚未提交；以下拟议范围保留为批准依据。

## 拟议范围

| 源文件 | 完整函数 | 归档位置 |
| --- | --- | --- |
| sched/slack_estim.py | build_score_dict_ref_flops，L24-28，共5行 | sched/slack_estim_unused.py |
| sched/slack_estim.py | get_chains_info，L410-416，共7行 | sched/slack_estim_unused.py |
| task/graph_breakdown.py | sort_chains_by_ddl_flops，L19-20，共2行 | task/graph_breakdown_unused.py |

三段共14行，逐字节移动，不改签名、函数体、原文件 imports 或其余空行。新归档只补必要导入；get_chains_info 继续借用活的 decompose_dag_into_chains。没有证据证明这些函数与现用版本等价，按独立未用能力归入 unused。

## 判断依据与边界

沿 main_approach → setup → task_cfg → slack_estim → graph_breakdown 追踪。164份 tracked/已批准新增源码及脚本中，三个名字只在定义处出现，没有调用、导入或相关星号导入。其余 tracked 文件仅旧 symbol-candidates 表有同名记录。仓外调用情况未知。

三个旧模块导出名会退休，已存的旧函数引用 pickle 将不能继续加载；这是明确的兼容边界，不能称为完全无行为变化。新路径函数应可 pickle 往返。活模块其余公共导出完整集合必须保持：slack 105→103，graph 6→5。

sort_chains_by_ddl_flops 对非空列表原有 IndexError；score 助手在中途失败前可能已部分写入；get_chains_info 不排序、不筛掉首尾节点。这些原行为均保留，不在清理时修复。

现用 get_chains、全部 slack 求解、绘图、图分解、graph_breakdown 自带测试和 main 块均保留。类内、保护路径、依赖、未跟踪文件不纳入本批。REQ-003及既有暂缓裁决不变。

## 基线与验收

基准 master，实际 HEAD a1d933b1f26efd4d569eb3d8ffb313447294443b。归档引用 archive/test_pipeline-20260612。基线 JSON 为 cleanup/reports/b19-unused-slack-graph-helpers-baseline.json。

- 原函数13组输入各执行两次，返回类型、异常和可变输入前后状态均已存档。
- gurobi 环境十套回归157项：152通过，原有5项 collector 失败，0 error/skip；日志 /tmp/scheduler-b19-preflight-kq_vvumu，30.27秒。
- 新 test_unused_slack_graph_helpers.py：先验证目标缺失的 RED，再测原行为、连续原块、签名、完整导出集合的两种导入顺序、旧/新 pickle 和共享图分解函数。
- 新增测试后合跑完整原十套加新文件；三个入口 --help；B13-B16收益探针；162份非本批源码全量哈希不变，不抽样。
- 原文件精确删除12行/2行，分别剩余 SHA bda2da01c0bb32873d021dd32efc0debb5017333221baa3bd5cfbf7f17cc39c5 / 486b438fa3325f8d035ef91ca63518649986e21aee74cb761092908c579cdbd1。
- graph 原无末尾换行。若 apply_patch 只多补一个 LF，提请批准：在原 SHA、完整 actual==expected+LF、预期无LF三项断言通过后，仅 truncate -s -1 还原工具副作用。其他差异立即暂停，绝不放宽期望值。
- CSV用解析器追加，保留全部历史前缀与既有坏行；状态与变更历史同步记录。

## 恢复与授权

实施后生成仅覆盖两源文件、两归档和新测试的五文件限定补丁，持久化压缩内容、SHA和提取/检查/恢复命令；先执行 git apply -R --check，不实际回退。禁止整文件 checkout 覆盖其他批次。

需真实 reviewer 批准提案后才实施，实施后再独立验收。未授权提交或推送。

## 执行结果

E0074已由真实reviewer批准，K2/K3口径随后由其独立澄清：slack减两个导出名、graph减一个；162非本批文件必须全部一致，无守卫差异豁免。其166源码计数多列了B12已迁走的两个根文件，已更正为实际存在的164份，不是B18计数时点差异。

三函数14行已原字节迁入两份同目录_unused。RED为1failed/28deselected，随后新增29项全过。原EOF由完整字节断言保护的truncate恢复，未改期望。

完整十一套186项：181通过、相同5项collector失败，0error/skip；29.23秒，日志/tmp/scheduler-b19-regression-ye0fkl_1。三入口help、B13-B16探针、162非本批哈希全量、151Python语法及保护区/依赖/HEAD/index检查通过。

CSV追加各3行且原字节前缀保留；move42行仅原8坏行，reachable264行全10列。五代码文件限定恢复patch已存JSON validation.recovery，reverse--check通过，尚未回退。

E0075记录实施完成；后经真实reviewer E0076验收及补核，现为ACCEPTED。未提交、未推送。

## 独立验收

Codex直接解析reviewer的/tmp/b19_review_junit.xml和/tmp/b19_new_only.xml：完整186=181passed+同5失败，新增29/29，均0error/skip，11模块数量匹配。reviewer补跑三help和B13-B16探针，并按本批20734/71003字节前缀及各3行增量重验账本；已更正原事件的旧数字和truncate未动用表述，不改原历史事件。

最终162非本批加5本批代码文件哈希全量保持，持久限定恢复patch再次reverse--check0未恢复。B19关闭，无在途请求。全仓语义审查仍未完成，原取消和暂缓事项不变。
