# B14 未用路径工具分离提案

状态：E0047批准后实施，真实reviewer在E0049独立复测后验收B14；CSV统计证据另经E0051勘误关闭。未提交、未推送。

## 范围

从utils.py分离三个完整函数到现有utils_unused.py：get_log_path_str、_normalize_path、check_paths_equal。包含软比较器专属标题，共21行；源文件只删除，目标文件只在现有1449字节之后追加必要import和原片段。既有HDF5归档原样保留。

这些是当前未接入调用链的工具，不是可替换当前路径体系的新接口。软比较返回bool，sim_main.compare_paths则严格比较并可能抛AssertionError；两者不能互换。目录getter和PathContext的日志文件getter也不能互换，因此归为unused，不声称已完成路径迁移。

## 保留结论

- build_path_old/get_csv_path_str/get_case_path_str仍经sim_main.build_paths_and_ctx被sim_main_old使用。
- get_cfg_n仍被run/cfg_parser.py使用；input_parser配置接口不改。
- pyinstr_profiler被mapper/mem_planner两个方法的装饰器使用，保留。
- PathContext类完整保留。prepare_induced_env_if_needed关联历史induced runtime行为，暂列REVIEW，不混入本批。
- Found类、重复numpy导入、core_distr/vectorized_core_allocation和所有现有测试不动。

## 基线与验收

执行前HEAD为a1d933b1f26efd4d569eb3d8ffb313447294443b，主线基准master。utils原SHA为14bcb37deccafb0b390ed8cb724eba4b0f167dee1aaa4e4d4c97b68168542965，预期剩余SHA为508f1366fb8bb08f16f1ad5f57e651692be7915c6a3c29e054ce73970dcd4e03。utils_unused原SHA为8f69a175635908b515cc84234ccae3ef0ce795933a6a81079822830fce6198fb。

已执行比较8场景和日志目录5场景，保留原输出/异常/输入处理；实施前后逐字段比较。新测试RED/GREEN后，联合B10/B11/B12/B13/collector回归；既有基线53通过、5项collector失败，不屏蔽失败。3入口help、153份其他源文件哈希、依赖和保护区零diff都需核验。复用两份账本，追加保留原列结构与历史字节。

归档后只有utils_unused提供这三个符号。不保留旧utils名称的导入或猴子补丁契约；仓外调用未知，这是reviewer须审核的边界，不将仓内无引用当成绝对无风险。

## 恢复边界

只反向本批21行分离与归档追加尾部，恢复到B13已验收状态。禁止从HEAD整文件恢复utils.py或共享记录；那会撤销B13及其他agent的记录。本批没有提交或推送，后续源码改动仍需另行提案。

## 实施结果

源utils相对B13仅删除21行，剩余SHA与预期相同；归档原1449字节完整保留。新测试先RED1失败，再GREEN19通过。六套联合回归72通过、5项原有collector失败，无error/skip；三入口正常，153份其他源/脚本哈希和依赖/保护区不变。当前140个Python全部AST可解析，不代表全部完成人工审查。

两份账本各追加一行，原前缀完整保留。move-ledger第2/3/4/14/15/19/20/21行存在历史列宽异常，与HEAD逐字段相同，未修；本批新行均符合14/10列结构。

E0051确认标准csv.reader全表解析的真实异常数为move-ledger 8条、reachable-files 0条。E0049曾报告12/44，原因是普通逗号拆分误算了引号内逗号；其历史原文保留，最终统计以E0051为准。没有因此修改合法CSV行，也没有修旧宽列行。

恢复补丁保存在JSON validation.recovery.forward_patch及/tmp/scheduler-b14-regression-pgo2vwm4/b14-source.patch。仅运行过以下第一条检查，未执行第二条恢复：

```bash
git apply --reverse --check /tmp/scheduler-b14-regression-pgo2vwm4/b14-source.patch
git apply --reverse /tmp/scheduler-b14-regression-pgo2vwm4/b14-source.patch
```

临时文件失效时，从机器报告提取同一forward_patch再检查。恢复只涉及B14两个源文件差异，不会从HEAD覆盖B13；测试和报告用本批限定补丁另行撤销，共享历史追加补偿记录。
