# B24 mapper 历史函数分离提案

状态：CLOSED / ACCEPTED，REQ-022，真实reviewer E0100及同会话补核完成。原迁移按E0094/E0096执行，新测试两行按E0098修正；新40项全过，未提交。以下提案与执行段保留历史时点，最终验收以文末为准。

E0096已确认的逐文件清单：old使用future annotations、typing List/Dict/Set及活Block/ContentionGroup/MemMap；unused使用future annotations及活Block/CyclicBlock。以下提案计划作为历史保留，实际执行和验证见文末。

## 范围

| 原位置 mapper/mem_planner.py | 同目录目标 | 行数 |
| --- | --- | --- |
| stat_overlapping_old，429–536 | mem_planner_old.py | 108 |
| scan_overlap_1d，712–721 | mem_planner_unused.py | 10 |
| sort_fn_conflict_s_r，801–802 | mem_planner_unused.py | 2 |

完整函数原字节移动，原文件纯删120行。保留全部7个类，包括seq_mapper_old与类内注释方法；保留其他13个顶层函数、imports、空行和原尾LF。剩余SHA固定为482af52853a9ca50ccfef4d940ac6f07bee92b9e4f87e170000031e258e39d10。归档仅补future annotations及类型/活数据类imports，不声明导入隔离。

## 依据与风险

已完整读mapper模块、空包初始化和保护测试，并沿其包装器/旧入口追踪。174代码及340份可读tracked/approved文本范围内，排除管理记录后，三个候选只有定义，无调用或保护文档API条目。当前stat_overlapping被scan_conflict、MemMap.seq_mapper调用，旧统计实现按old分类；一维检查和冲突排序不是现函数的等价替代，按unused分类。

现有test_mem_planner明确导入六个保留接口，不能整包归档；但它目前导入即报NameError: default_binpack_cfg，不声称该旧测试链能运行，也不顺带修复。类内、手工诊断和旧引用全部保留。

三个旧导出和函数引用pickle路径退休是MEDIUM兼容变化，公共名全集107→104；仓外调用未知，不加兼容层。须由真实reviewer明确批准，不能由“零引用”推导删除授权。

## 修改前证据

日志：/tmp/scheduler-b24-preflight-16n_tbiw。两个独立gurobi进程结果逐字段相同：
- 旧统计13组、一维检查7组、排序7组；包含真实MemMap的first/best/worst三策略、输入排序/耗尽、异常、部分状态和stdout。
- 1组跨调用默认容器累积，3组现用scan_conflict/seq_mapper/priority_mapper真实路径。
- scan_overlap_1d无重叠返回None；旧函数可变默认值会累积。这些是要保留的原行为，不“修正”为更合理的结果。
- 保护test_mem_planner导入失败已单独记录，不纳入“通过”口径。

## 实施门槛

批准后先新建test_mapper_legacy_helpers.py，观察归档目标不存在的AssertionError RED，再移动原块。计划40项新测试，普通行为每组双跑，可变默认值与完整公共名/签名检查采用fresh子进程，覆盖旧pickle失败、新位置往返及借用类的身份。测试锁定原行为baseline SHA，不从新实现计算期望。

与既有14模块合跑，预期330=325通过+同5个collector失败；以实际XML为准，不改/跳过原失败。复跑main/motiv/abla help和保护测试导入错误。173非本批SHA无豁免，保护区/依赖/HEAD/index不变。

move追加3×14、reachable追加3×10，保留原25708/74515字节前缀和8条历史坏行。四代码路径限定恢复patch需持久化，并先reverse --check，不整文件checkout。

其他偏差暂停。申请仅在实际字节恰为预期±1LF、原SHA/预期SHA/完整bytes三断言全过时纠正工具EOF误差；不得改预期。实施后RESULT仍须独立ACCEPTED。无commit/push授权。

详情：cleanup/reports/b24-mapper-legacy-helpers.json。全仓语义审查尚未完成。

## 实际执行结果

原源纯删120行，完整剩余SHA482af528…e39d10；三个原块连续且各一次，imports按E0096。工具丢1尾LF已按O5三断言补回并fsync，未改预期。7类内部和其余源码保留。

RED为目标不存在的1failed/39deselected。首轮新40中39通过、1类型断言失败，已按O5暂停并由真实E0098批准仅修正新测试两行；原始与归档都是Optional[MemMap]。这次错误尝试保留于validation_attempt_1，未改生产或行为golden。

最新日志/tmp/scheduler-b24-validation-b199bqnc：新40全过；十五模块330=325通过+原5同名同异常collector失败，0error/skip，14个既有模块数量逐项一致。main/motiv/abla help全0；保护test_mem_planner仍在28行缺default_binpack_cfg、rc1，不修、不计通过。全仓测试仍有已知问题。

173非本批SHA无豁免；迁移后177代码SHA再核一致，161份Python可解析。保护区、现有测试、依赖、HEAD/index不变。git diff --check仍只报原sim_main.py:677尾空行警告。

move50→53新增3×14，reachable271→274新增3×10；原25708/74515B前缀保留，8旧坏行不修。B24 JSON validation.recovery保存四代码路径限定patch，SHA577a2834…b5d9ebe，持久提取后reverse--check0，未恢复。

NoC只读跟进：model/noc.py导入TypeError: abstract class，原类和文件保留，未顺带修改。等待真实reviewer验收E0099，不开始下一源码批次、不提交推送。


## 最终验收

真实reviewer E0100已ACCEPTED。Codex直接读取独立XML：b24only.xml为40通过；fifteen.xml为330=325通过+相同5个collector失败，0error/skip，15模块计数及异常消息一致。

reviewer补核文件/tmp/b24_review_final/supplement.json（SHAac7f5d2b…be92da）已持久写入B24报告reviewer_acceptance：173非本批文件按迁移前SHA逐键一致；两表原前缀、53/274总行及8坏行保留；4条实际CLI命令rc0/0/0/1；三个归档函数pickle protocol4往返身份全True。177份迁后快照检查与173份迁前后检查分别成立，模块导入不替代pickle证明。

源120行原块迁移和剩余字节重建、3×14/3×10账本追加、四路径限定恢复patch均再次核实。收尾只更新记录，无源码变更或提交。B11-B21+B23+B24共13个源码批次已验收未提交，B22只分析。NoC仍REVIEW保留；全仓语义审查尚未完成，下一源码批次仍须另提案。
