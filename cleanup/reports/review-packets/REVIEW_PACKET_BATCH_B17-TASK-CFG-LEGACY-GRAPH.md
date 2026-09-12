# B17：task_cfg 历史图生成分离

状态：ACCEPTED，REQ-015已由真实reviewer在E0064独立验收并关闭。原样分离、18项新测试、联合127passed/5原有failed、三入口、字节重建、账本与恢复检查通过。未提交、未推送。

审核勘误：E0062已由reviewer直接读old/allocator_agent.py L561/563/577/580确认两保留函数确有借用，撤回E0061(a)的错误更正。old_num_hp位于approach_initiator.py，本提案从未将它纳入task_cfg迁移。验收范围为tracked/index加明确批准新文件，不采用未过滤os.walk清单。

## 范围与理由

沿 main_approach → approach_sched/collector → approach_initiator → task.task_cfg 深度优先追踪后，提出同一模块的两段历史分离，不扩展为算法重构。

| decision | 原位置 | 动作 |
| --- | --- | --- |
| B17-MOVE-001 | task/task_cfg.py L24-82，59行旧 vis_task_static_timeline 注释实现 | 原样移入 task/task_cfg_old.py；同名现用绘图函数及调用保持 |
| B17-MOVE-002 | task/task_cfg.py L551-697，147行 creat_jobTask_graph | 完整原样移入同一归档，包含 return 后仍属于函数的历史说明 |

两段合计206行。选择 old：旧图展开职责已由当前 creat_logical_graph/creat_physical_graph/gen_workloads 路径承担；旧绘图注释也有同名现用实现。不是声称新旧图生成结果等价。

## 已有证据

- 主线master；实际HEAD a1d933b1f26efd4d569eb3d8ffb313447294443b。原文件SHA 4320789e8a8f2d5c9e37333bf8b0ee544154a68119d517004770d62ad8521a7d。
- 预计仅删除上述206行后，剩余完整文件SHA为6194c815e202c298f7789bb37be929dcc59f19d513a2ebb9a476a2fc3b0ef14b。
- 144个明确范围Python文件无旧函数执行或导入引用；17处task_cfg导入均为按名导入，不借用候选，也无该模块的star import。tracked文本仅有定义、输出文件名及L1356的注释调用。
- 旧函数模块依赖仅nx/pd/np/plt及内置函数，注解需Dict/List。归档补必要import，不从活task_cfg重导入旧函数。
- 11组真实CSV基线已执行：空图、控制边、单任务、等比/上采样/下采样、多副本、非Entry源、零factor、非整数thread和真实PDF输出。
- 原行为保留：Exit只接最后生成的副本、零factor的ZeroDivisionError、非整数thread的TypeError。plot=True还会给返回图添加layer属性，测试必须比较这些属性。
- 当前公共导出119个，提案退休一个旧函数名后应为118个；完整名字和描述已保存，不能只抽样。
- 修改前八套联合114项：109passed、相同5个原有collector失败，无error/skip。日志 /tmp/scheduler-b17-preflight-tests-fev32mde。

## 明确保留

现用vis_task_static_timeline、全部现用图构建与JSON I/O、全部import和剩余源码原样；load_taskint/redist_ert_dll仍被old/allocator_agent借用，不能连带搬走。类内方法、既有测试、保护文档、依赖和此前B11-B16成果不动。approach_initiator.old_num_hp仅REVIEW，不混入本批。

归档后的调用路径为task.task_cfg_old.creat_jobTask_graph。旧task.task_cfg下的函数名及旧函数引用pickle不再承诺兼容；函数签名、返回值和实现不改，不新增兼容壳。这一退休边界需reviewer明确批准。

## 执行与验收

1. 真实reviewer批准后，新增test_old_task_cfg.py，先确认“归档不存在”的RED。
2. 两段原样分离；活文件0增加/206删除。验证完整剩余字节和两原块，各只出现一次。
3. 新测试逐项比较11组原结果，覆盖实际PDF生成、图节点/边及属性、异常；归档独立导入不加载task.task_cfg。
4. fresh进程核对119→118完整导出与剩余描述；随后合跑既有八套和新测试，失败集合仍只允许原5项；三入口help通过。
5. 核对159份非本批源码/脚本SHA不变；CSV只追加本批move/reachable行，保留历史字节和schema；同步全局记录后发送RESULT，等独立验收。

## 恢复

已执行结果：九套132项=127passed+5原有collector失败，无error/skip；3help通过，B13/B14探针通过。完整119→118导出、11组原图/PDF/异常、原注解均通过。源0增/206删且完整字节重建相等，159非本批SHA不变，146Python语法通过，保护区/依赖零diff。

CSV已追加move2行14列/reachable2行10列，旧18839/69612字节前缀保留；move仅8历史异常不修，reachable259行全10列。限定恢复补丁已存JSON validation.recovery（zlib+base64、原补丁SHA及三个命令），从持久报告解码后的git apply -R --check通过，未执行实际恢复。联合日志/tmp/scheduler-b17-regression-_sfebunx。

归档引用archive/test_pipeline-20260612；恢复使用本批实际写前字节生成的限定补丁，并先git apply -R --check。只撤销task/task_cfg.py本批两个删除块及本批新归档/测试，不整文件checkout共享未提交内容。B11-B16保持未提交，本批不含提交或推送授权。

机器证据：cleanup/reports/b17-task-cfg-legacy-graph-baseline.json。
相邻只读审查：cleanup/reports/dfs-approach-runtime-review-20260906.json。
