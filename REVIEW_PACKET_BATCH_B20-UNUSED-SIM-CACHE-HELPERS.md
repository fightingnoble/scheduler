# B20：未用的仿真缓存助手

状态：CLOSED / ACCEPTED，REQ-018，真实reviewer E0082独立验收通过。迁移获E0078批准，EOF单字节恢复获E0080批准。Codex已直接读取reviewer的/tmp/b20_review_final/twelve.xml（224=219+原5失败）和b20only.xml（38全过），并复核全部169代码哈希未变。

## 本批改动

四个完整函数从sim_main.py原样迁入同目录sim_main_unused.py：

| 函数 | 原行号 | 行数 |
| --- | --- | --- |
| ensure_csv | 84-88 | 5 |
| prepare_induced_env_if_needed | 172-193 | 22 |
| get_core_num_from_trace_name | 705-730 | 26 |
| check_max_bin_num | 732-752 | 21 |

共74行，函数内容、签名及prepare→get_core内部调用保持。归档补充os/re、PathContext、两个路径模板、load_pickle，以及活compare_paths/generate_bin_paths的导入；不声称归档独立于sim_main。原其余16个函数、imports、常量和空白保留。类内、perform_bin_packing、旧入口、保护文档和现有测试不动。

沿approach_setup→sim_main审查全部20个顶层函数；167份明确源码/脚本中，候选没有外部可执行调用或星号导入。历史开发记录、路径替换说明和注释仍保留。B14当时的范围排除已在E0078独立重评，并非本批自动获准。

## 验证结果

- 先RED：缺少归档文件导致1failed/37deselected，/tmp/scheduler-b20-red.log。
- 新增38项测试单跑全过，/tmp/scheduler-b20-green.xml。18组真实CSV/缓存/pickle I/O各双跑；比较返回值、stdout、文件字节、PathContext/trace/plot状态及原异常。
- 完整十二套224项：219passed，5个与基线同名同异常的collector失败，0error/skip，29.94秒。日志/tmp/scheduler-b20-regression-lomypf0o/pytest.xml及pytest.log。
- 三入口help和B13-B17收益探针本轮实际运行通过，日志在同目录。166非本批源码/脚本SHA全部原样，153Python可解析，保护区/依赖/HEAD/index未变。
- 公共名完整集合121→117，恰少四个迁出函数，fresh双导入顺序通过。旧函数pickle加载失败、新位置pickle身份往返均已测试。
- 源0增/74删，剩余30583字节，固定SHA：1d04e4aa038b63bc289834fc350442d9ff2b929faabb42a042deb7803bc86520。归档四原块各连续出现一次。

## 明确的风险和检查警告

四个旧sim_main导出及函数引用pickle路径退休是真实兼容变化，仓外使用未知；本批不提供旧名转发，不声称无兼容影响。空缓存原UnboundLocalError、缺目录StopIteration、坏pickle转FileNotFoundError、路径断言失败及失败前部分状态修改均保留，没有修复。

工具曾丢失末尾一个LF，已暂停并通过E0079→E0080单独批准后恢复：原SHA/当前SHA/长度/完整actual+LF==expected断言，r+b仅写1字节并flush/fsync，再全字节核对。未修改期望。

Git仍给出唯一警告：sim_main.py:677 new blank line at EOF，git diff --check rc2。两空行是HEAD原704、731行，迁移后成为末尾；不删除原空白消警告。临时验证器首次因此停止，查清后仅改为精确记录该警告并证明原字节来源，再完整重跑。代码、测试、行为基线均未为通过检查而修改。独立reviewer在E0082明确接受该警告和处理方法，不写成“diff-check通过”。

## 账本与恢复

基准master；实际HEAD a1d933b1f26efd4d569eb3d8ffb313447294443b；archive/test_pipeline-20260612。

已核验追加4条14列迁移记录和2条10列分类记录。原22038/72055字节前缀SHA完整保留；move42→46且8条历史异常原样，reachable264→266且全表10列。

机器报告：cleanup/reports/b20-unused-sim-cache-helpers-baseline.json。validation.recovery保存三个代码文件限定压缩patch、SHA以及extract/check/restore命令。已从持久报告解码核对SHA和三个路径，reverse--check为0，未实际恢复。恢复须显式授权后先check，禁止整文件checkout。

当前没有提交或推送授权。全仓语义审查尚未完成；REQ-003取消项和REQ-005/006/007暂缓项不重开。
