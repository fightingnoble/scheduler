# B21：未用的消息读取助手

状态：CLOSED / ACCEPTED（真实reviewer E0086），REQ-019。Codex已直接核对独立测试日志及验收后全部代码SHA。以下提案/执行/RESULT段保留时间顺序，以本状态和末尾验收段为准。

## 拟议范围

将model/message/msg_dispatcher.py中的msg_read（30-48，19行）、msg_filter（50-51，2行）原样移到同目录msg_dispatcher_unused.py，共21行。归档仅补from model.buffer import Data；内部msg_read→msg_filter随原字节保留。

MsgDispatcher类、Queue/Data导入、所有注释和剩余空白保持原样。两个助手是当前未用的独立能力，用unused而非声称被新实现等价替代。没有删除代码，也不清理类内方法。

## 引用判断

169份明确源码/脚本和tracked文本中，两个函数无外部可执行调用；11处相关导入均只取MsgDispatcher类。msg_filter在旧sched_fn和scheduling_table_event中的命中是关键字参数或局部过滤字典，不是该函数。动态导入和属性查找点已逐项查看，未发现调用链；仓外使用未知。

邻接模块保留：message_handler四函数有旧allocator或Repack的直接/传递调用；scheduling_table_event七函数被runtime_legacy使用；MessagePipe虽未发现类名的外部Name调用，old/allocator_agent.py:8仍有导入，不能据此整体移走。旧runtime和类内均不动。

## 行为与边界

21组案例各双跑：子串匹配、重复消息、空关键字、容器成员关系、原TypeError；空队列/无消费者、状态标志、广播输出、多个前驱/进程、缺源、频率为零、缓冲区容量拒绝，以及失败前已写入的数据和状态。

使用真实Queue/Buffer/Data；进程和任务输入用只包含所需字段的SimpleNamespace记录。原行为包括先清空队列、Buffer.put返回False被忽略、缺源/除零前pred_data已修改，全部保留不修复。另记录MsgDispatcher默认队列和外部队列两组实际广播/定点发送基线。

旧模块的msg_read/msg_filter导出和原函数pickle路径将退休，属于明确兼容变化，不提供旧名转发。其他公共集合5→3恰失这两个名；新归档函数保留pickle身份往返。

## 验证计划

基准master；实际HEAD a1d933b1f26efd4d569eb3d8ffb313447294443b；archive/test_pipeline-20260612。现有十二套224=219passed+原5collector失败，0error/skip，36.33秒，日志/tmp/scheduler-b21-preflight-406edjxl。

拟新增35项测试，先RED后迁移，再GREEN；十三套应259=254passed+相同5失败。另实际跑3help和B13-B20保持性探针，168非本批SHA全量无豁免，所有既有测试/保护区/依赖/HEAD/index保留。

源只能0增/21删，固定剩余SHA0c9df6adcec6cdda83d285d7e865097e74b9103f513eeadb97f2ac4a3ac3fbcd，原文件没有EOF换行。若工具仅多加或少留一个LF，申请以原SHA/预期SHA/完整字节等式断言保护下，仅truncate一个多余LF或r+b补回一个缺LF并fsync；其他差异即暂停，不改期望、不整理格式。B20已验收的sim_main.py:677空行警告应与本次基线rc2和原文完全一致，不以删除原空白消除。

## 账本与恢复

账本本批基线move23862字节、reachable72773字节；拟新增2条14列迁移行和2条10列分类行，原前缀和8条历史坏行保留。JSON：cleanup/reports/b21-unused-message-helpers-baseline.json。

恢复仅三个代码文件：原源文件、归档和新测试。实施后保存限定压缩patch、SHA及提取/检查/恢复命令，reverse--check但不实际恢复；禁止整文件checkout覆盖其他批次。

需真实reviewer先批准提案，实施后再验收。无提交、推送授权。

## 实施记录

按E0084 M1-M6完成两函数共21行原字节迁移。源0增/21删、1717字节、剩余SHA与预计算完全相同。工具加上的一个EOF LF已按M5完整字节断言保护移除并fsync，原无尾LF得到保留。RED1failed/34deselected后GREEN35passed，日志/tmp/scheduler-b21-green.xml和/tmp/scheduler-b21-green.log；完整回归及RESULT尚待完成。上文拟议和计划为获批时原记录，不代表当前仍未实施。

## 回归与恢复结果

十三套259=254passed+原5同名同异常collector失败，0error/skip，38.93秒。日志/tmp/scheduler-b21-regression-mnj4yb7z/pytest.xml和pytest.log；3入口help及B13-B20探针实际通过，同目录*.help.log与benefits.log。168非本批SHA全量原样无豁免，155Python AST可解析，保护区/依赖/HEAD/index不变。git diff --check保留B20既有唯一rc2警告，全文与基线一致。

两账本追加2×14和2×10，写后复核待完成。JSON validation.recovery已存三代码路径限定压缩patch、SHA5315ace77f112cef6715cfb760ff8f00a9da844cc6017a508e8bdaee6025b1f9及extract/check/restore命令。临时patch reverse--check0，下一步从持久JSON解码再检查，不执行恢复。

## RESULT交接

写后核验通过：move46→48新增2×14，reachable266→268新增2×10，23862/72773B原前缀SHA完整保留，8条旧坏行未修。由持久JSON解码限定恢复patch，SHA及三路径一致，reverse--check0，未实际恢复；全部171代码SHA后置核验一致。E0085已交真实reviewer，验收前不继续下一源码请求。

## 独立验收关闭

真实reviewer E0086接受M1-M6。Codex直接读取/tmp/b21_review_final/b21only.xml（35passed）和thirteen.xml（259=254passed+原5同名同异常collector失败，0error/skip），并核对十三模块计数、3help/probes日志。168非本批SHA全量无豁免；验收后全171代码SHA仍一致。持久恢复patch和账本前缀/新列检查均通过，不执行恢复。

当前无在途请求。B11-B21共十一批未提交推送，全仓语义审查未完成；下一源码动作仍须单独批准。两旧函数导出和pickle路径真实退休，仓外使用未知；不为修旧缺陷或消B20 EOF警告改代码。
