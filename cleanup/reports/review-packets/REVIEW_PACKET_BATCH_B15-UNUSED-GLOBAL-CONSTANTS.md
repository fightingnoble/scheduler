# B15 未用全局成本参数

状态：REQ-013已关闭。E0053批准→E0054 RESULT→真实reviewer E0055 ACCEPTED。未提交、未推送。

## 执行范围

将global_var.py中的十个完整赋值行原样移到同目录global_var_unused.py：

- overhead_pushpull_per_core
- overhead_of_enqueuing_op
- overhead_of_dequeuing_op
- clock_period
- SRAM_size_per_core
- GLB_BUFFER_SIZE
- MIN_CORE_NUM
- W_perc
- A_perc
- O_perc

共10行；不整理空白、不改表达式或内联注释。归档保留40E6、1/3等原写法，不引入新依赖。主线基准master，执行HEAD为a1d933b1f26efd4d569eb3d8ffb313447294443b，源SHA5a66c7836d03f4a50251c8cee3a8533b48a63ebb69bb1606df8b59577108562d，预期纯删除后1eae1477f532e916e9829942bc4474bc79604809d85171a14356d99818d769cd。

## 保留判断

140个Python的引用提示不是直接删除依据。加入内部依赖和文档证据后，保留精度常量整组（包含规范仍定义的零代码引用bits）、所有路径格式和目录、trace_file/list、math/os、两个数值函数及lambda、现用资源性能参数和算法配置。elim_error在resource_agent内真实调用；elim_nume_error经utils转导出到approach；目录和routing_table_fn_fmt仍是活格式的内部依赖。

十个候选没有外部代码名字/属性/显式import使用，也不参与global_var其他表达式；全仓tracked文档/配置/脚本无完整单词命中。分类为unused，未声称有等价的新版本。

## 风险与验证

51个导入站点中31个为星号导入，因此不能省略命名空间验证。迁移后不再提供十个旧global_var或utils导出名，仓外使用未知，不承诺兼容；保留其余公共名字、数值和路径接口。fresh process检查真实global_var和utils，兼顾Python星号绑定；不用给活模块加兼容壳来掩盖边界。

新测试先RED后GREEN，原赋值行与十个值/类型逐项验证；联合现有B10-B14及collector，72通过+5原有失败作为基线，三入口help和155份非本批源/脚本哈希不变。依赖、保护区和既有测试不动。账本只追加正确14/10列新行、保留所有前缀，不修旧8条格式异常。

## 实测结果

- global_var.py纯删除10行，剩余SHA与提案预计算完全一致；十条原赋值各在归档出现一次，没有函数、接口或类内修改。
- RED为1failed/12deselected（目标未创建），GREEN为13passed。fresh process实测global_var、utils、model.resource_agent恰只少十名；其余67个共享global_var导出的对象身份和类型/值描述不变。
- 七套联合90项：85passed、5个原有collector失败，0error/skip。五个失败名/类型与B14基线相同；三入口help均rc0。155份非本批源码/脚本原样，142份Python语法通过，requirement和保护区零diff。
- CSV标准解析验证新增1行14列及2行10列，既有字节前缀全保留；move仍只有8条历史坏行，reachable无异常，未修历史。
- 日志：/tmp/scheduler-b15-regression-p_no7rv9/。机器报告记录全部基线、实测值、字节证据及恢复命令。

## 限定恢复

cleanup/reports/b15-unused-global-constants-baseline.json中的validation.recovery包含完整forward_patch及check_command/restore_command。补丁仅还原本批global_var十行并移除两个准确新增代码文件，不覆盖B11-B14或共享记录。git apply --reverse --check已通过，未执行恢复。恢复必须先检查当前diff；共享状态、历史和账本追加补偿条目，不整文件checkout。

reviewer已独立复测并验收E0053 G1-G6，又用两个fresh进程重演迁移前后全量导出集合。global_var 77→67、utils 107→97、model.resource_agent 93→83，各恰少十名，无新增或保留值描述变化；Codex逐字段复核/tmp/b15_ns_snap/pre.json、post.json，也自行重演得到相同结果。E0055的五名抽查只是早期过程记录，最终以完整集合证据为准。REQ-013关闭，暂缓的Repack和类内清理不重启。
