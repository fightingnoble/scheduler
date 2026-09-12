# B18：分离未使用的周期事件生成器

状态：ACCEPTED，真实 reviewer E0072 已验收；完整十套及恢复检查补核已完成，未提交。

## 范围

沿 task_cfg → task_agent → resource_agent/e2e_latency 的实际依赖阅读。只迁移 model/event_gen/e2e_latency.py 的两个完整函数：

| Decision | 原位置 | 目标 | 行数 |
| --- | --- | --- | --- |
| B18-MOVE-001 | naive_period_event_gen，L7-11 | 同目录 e2e_latency_unused.py | 5 |
| B18-MOVE-002 | e2e_var_sim，L14-34 | 同目录 e2e_latency_unused.py | 21 |

两段原字节完整保留，源文件只删26行，不改其余写法、imports、签名或函数体。归档补齐 future annotations、Dict/Union/numpy，并从原模块导入仍活跃的 jitter_gen_biside；不复制其数值实现。目标两文件目前均不存在。

选择 unused：两个生成器是当前未调用的独立能力，未证明它们与现用离散事件生成器等价，不据此宣称“旧新实现可互换”。

## 为什么只移这两个

146份Python的名字/属性/精确字符串和导入检查未发现两候选的代码调用者；tracked文档和脚本交叉检查仍发现2023年历史记录 doc/dev/change_log_2023.md:676，应保留并披露，不能把它说成无任何引用。

discrete_event_sim 被 sim_main_old.py:174/180 使用，其整数生成器 get_intger_gen 也保留。jitter_gen_biside、exp_jitter 和参数助手被任务类、资源类、旧 Scheduler、data_pipe 和 optimizer 借用，都留在原模块。trace_analyser 由 analyze/analyze_timing.py 实际调用，performance 四个活接口也保留；类内方法不清理。

e2e_var_sim_en/e2e_var_sim_para 是配置名，不是待迁函数的调用证据，相关CLI与文档不改。仓外消费者未知。

## 明确的兼容边界

旧 model.event_gen.e2e_latency 下两个函数名和对应旧函数引用 pickle 路径退休，不加原模块兼容导出；新入口为 model.event_gen.e2e_latency_unused。签名、默认值、返回和历史异常不改。归档有意借用活模块 jitter_gen_biside，因此归档导入会加载活模块；不声称二者导入隔离。

## 基线与验证

- 执行前 HEAD a1d933b1f26efd4d569eb3d8ffb313447294443b；主线master；archive/test_pipeline-20260612。
- 原源SHA 7c460e9258b1b7384ffb4605167e4ccc69b026fc83eb0161429bd9f60b994248。
- 预计算剩余SHA 3043c818e36dbdfcee8ee50573c5274a8549fbb1413e4cb55cde07cc0e990a7e。
- 原函数13组样本各重复两次一致：偏移参数被忽略、正/零/负周期、scalar/vector、固定seed、零/负事件范围、缺少scale异常和零scale；保留yield类型/形状、inf哨兵及StopIteration.value。
- 修改前九套132项=127passed+5原有collector失败，errors/skipped=0；日志 /tmp/scheduler-b18-preflight-dkteud1z。
- 批准后先建 test_unused_e2e_event_generators.py，观察归档缺失的断言失败，再做纯移动。新测覆盖上述行为、两个字节块、完整公共名81→79、双fresh导入顺序、签名及借用同一活helper。
- 合跑九套既有测试及新测，比较原5项失败；三个入口help；161份非本批源码/脚本哈希全部不变；保护区/requirement不动。仅有语法或help通过不足以说明完整实验正确。

## 记录与恢复

复用move/reachable两账本，csv.reader核列后追加，原字节前缀保持，不修8条历史坏行。双全局记录随动作更新；本提案阶段不追加执行账本。

仅本批三个代码文件生成限定恢复patch，并将压缩内容/SHA及提取/check/restore命令写入机器报告；执行 git apply -R --check，不实际回退，不整文件checkout。共享记录用限定差异或追加更正恢复。没有提交/推送授权。

机器报告：cleanup/reports/b18-unused-e2e-event-generators-baseline.json。

## 执行结果

两函数26行已原样归档，25新测试通过；完整157项=152passed+原5failed，三入口和B13-B16探针通过。完整公共名81→79恰失两名，13组每组双跑行为及四个pickle边界通过。日志/tmp/scheduler-b18-regression-oq0al2g0。

E0067纠正了reviewer原先混淆函数引用/生成器实例的判断：旧函数引用pickle退休是真实兼容变化。E0068单列reviewer自己的守卫白名单变化：原161基线保留，全量160不变+1已归因，不宣称161未变。

apply_patch曾给无尾换行的源文件多加LF。字节门捕获后按E0069暂停，E0070批准仅机械去掉1字节；现在完整重建、原EOF及预期SHA都吻合，diff0增/26删。没有放宽检查。账本原前缀保持，限定恢复patch已从持久报告解码并reverse--check通过，未实际回退。

E0072验收文字的“24/25”及回归范围已补清。Codex直接读取reviewer XML确认新增25/25、完整十套157=152passed+5相同collector失败，包含初次漏跑后补齐的B12七项；恢复补丁也独立校验通过。当前无在途请求，后续源码动作另提案。
