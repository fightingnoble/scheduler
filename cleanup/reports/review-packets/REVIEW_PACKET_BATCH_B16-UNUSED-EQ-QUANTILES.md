# B16 两个未用分位数函数

状态：REQ-014已关闭，真实reviewer在E0059独立验收B16并正式更正H1哈希笔误。未提交推送。

## 范围

沿main_approach→approach_sched→utils/global_var返回approach_Eq，先读完其依赖sched/ref_alloc_search。后者的全部辅助函数服务现用find_legal，主入口的标量/广播/逐元素断言也已实跑通过，整文件保留，不能误当废弃示例。

本批只将approach/approach_Eq.py的两个完整函数移到同目录approach/approach_Eq_unused.py：norm_inv_cdf（L42-43，2行）及exp_quantile（L45-51，7行）。源文件纯删9行，不整理空白或imports；归档仅补简短说明、import math及from scipy.stats import norm as _scipy_norm，原签名、函数体、docstring不变。math与_scipy_norm仍被活函数/分布类使用，原imports不删，不改依赖。

## 依据与保留

当前明确范围142Python中，两个候选无AST名字/属性使用、无其他源码文本引用；全部tracked文本精确单词搜索无命中。21个Eq显式import站点均不引用它们，也无Eq星号站点。模块内分布类直接用SciPy或discrete_quantile，不能把两候选说成已被新方法等价替换；分类unused。

find_legal虽然没有在Eq内调用，task_agent和chain_slack_assign仍通过Eq导入，必须保留。所有分布类及其方法、精度/时间工具、工厂和现用采样闭包均保留。time_eq/time_add/time_sub还有活显式导入和规范定义。B11现有32项兼容测试不改，其旧pickle样本不含这两个候选。

## 边界

旧approach_Eq与approach.approach_Eq中的两个函数名一起退休，不提供兼容重导出；外部导入、反射、monkeypatch及这两个旧函数引用的pickle不承诺兼容。归档新路径为approach.approach_Eq_unused，不新增根壳。其余43个公共导出及所有B11根/包别名身份保持。

包内实现仍是B11已批准但未提交的新文件，不能用git diff为空来声称没有改动。当前字节与HEAD:approach_Eq.py原实现一致，SHA为f5eff8c44b804be60e8de8a4c2d1a3afad9a127b9e80b931fd5edd86d5d0a773；预计算纯删后bb2e65224f12fe2555c2f12567276a110d8d409808a180f4d4282b86a289cdc8。根approach_Eq.py兼容壳不动，主线master，实际HEAD a1d933b1f26efd4d569eb3d8ffb313447294443b。

## 验证与恢复

18组迁移前真实函数结果已记录，包含norm端点/越界的inf/nan、exp无效p原ValueError及允许的零/负scale，全部保留，不借清理修正数学行为。审批后新增测试先RED再GREEN，覆盖原片段字节、18组结果、双导入顺序的完整45→43公共名/描述和别名身份、独立归档导入。

## 执行结果

- 源包文件与已记录的B11字节基准比较恰0新增/9删除，剩余SHA与预计算一致；两完整函数各在归档出现一次。原root alias不变，SHA6a869b99...c09e。既有B11报告保留为历史证据，不更新成当前源码快照。
- RED1failed/23deselected（归档缺失）；GREEN24passed。18组值/异常逐字段不变，fresh双导入顺序45→43恰失两名，无其他缺失、新增或描述变化；根/包模块及对象身份、find_legal重导出、共享time_unit状态保持。
- 八套联合114项：109passed、5个原有collector失败、0error/skip，失败名和类型与B15相同；三个入口help通过。ref_alloc_search原main rc0、stderr空、stdout SHA仍为bc9862ec...9b2b，输出与基线逐字节相同。
- 157份非本批源码/脚本SHA不变，144Python AST全部可解析，requirement/保护区/既有测试零diff。CSV新增1×14及2×10正确，原前缀保留；move仍只有8条历史异常，reachable无异常，不修旧行。
- 日志/tmp/scheduler-b16-regression-5t8gb_ze/；所有测试进程已完成，没有为历史失败加skip/xfail。

## 校验说明

E0057 H1的SHA正文误多一个2（65字符）。reviewer已在同一会话独立重算，确认正确64位bb2e65224f12fe2555c2f12567276a110d8d409808a180f4d4282b86a289cdc8；请在验收事件正式补记，Codex未改其历史。

第一次临时校验脚本在业务测试、help、ref_alloc及源SHA均通过后停在SequenceMatcher操作类别断言。默认auto-junk将两函数之间保留的空行也纳入替换块。实际文件等于原字节精确移除批准9行后的结果；关闭启发式匹配得到2+7行纯删除，Git最小差分也为0增/9删。只修正临时验证方法，源码和测试不动，修正后完整流程已重新跑通。

## 限定恢复

B16 JSON validation.recovery持久化完整补丁及check_command/restore_command，临时副本见其中temporary_patch。补丁仅恢复包内Eq本批9行并移除两个准确新增代码文件，不碰根壳或B11-B15其他成果；git apply --reverse --check已通过，尚未执行恢复。共享状态、历史和CSV只能追加补偿记录，不整文件checkout。

reviewer已在E0059独立验收H1-H6并正式补记H1笔误。随后双方还核对H2口径：排除所有下划线前缀的公共名为45→43；只排除双下划线为49→47，差集恰为_scipy_expon、_scipy_norm、_scipy_truncexpon、_scipy_truncnorm。旧注中math/find_legal/annotations的猜测已由reviewer在双全局记录更正；本报告统一采用45→43。原验收有效，无源码或测试追加修改，不重启取消/暂缓的Repack与类内清理。
