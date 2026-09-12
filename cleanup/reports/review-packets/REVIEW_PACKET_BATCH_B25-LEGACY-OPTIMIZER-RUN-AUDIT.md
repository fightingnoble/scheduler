# B25 独立实验与运行脚本审查

状态：CLOSED / ACCEPTED，REQ-023，真实reviewer E0102确认KEEP_AS_IS。本批只读分析，不构成迁移、删除或接口退役授权。以下申请段保留历史时点，最终证据见文末。

## 判断

| 范围 | 判定 | 依据 |
| --- | --- | --- |
| optimizer/scheduler_base.py | KEEP | 保护指南仍列出 draw_computational_graph；delta_ver 主程序实际调用它 |
| optimizer/ops_test.py | KEEP | 已有交互测试，包含图形、滑块和回调，不能按无人 import 判废 |
| optimizer 其余三个模块 | REVIEW，原位保留 | 概率调度实验内部互相依赖，缺环境不等于废弃 |
| old/ 四个已有归档 | REUSE_CANDIDATE，整体保留 | 不再拆其类、函数或历史演示；只查归档边界 |
| model/noc.py | REVIEW，原位保留 | 既有 TypeError 尚未进入修复设计，类内先不管 |
| run/cfg_parser.py、两份现行实验 wrapper | KEEP | 有明确 shell/CLI 调用，不由主入口 import 并不意味着无用 |
| run 其余扫描和清理脚本 | REVIEW，原位保留 | 旧调用路径、参数契约和写盘/删除副作用尚未验证，不能直接搬移或重写 |

保护指南的弃用范围是 optimizer/old/，不能扩大为整个 optimizer/。build_cat_prob_tensor 虽仅找到定义，仍与这套独立实验保留；当前无法取得其完整行为基线，不为减少函数数量另切一刀。

## 实际验证

完整读取7个Python模块及15个shell脚本；另一shell文件与已读版本作全文差异比较，确有26行FIFO演示，不是字节相同的副本。4个旧归档只核AST/import/header，不宣称全文审查；NoC沿用B24全文证据，本轮复核SHA和真实导入。共28文件。

16个shell均通过bash -n，仅证明语法。5个Python帮助入口退出0；motiv shell的-h按原usage函数退出1，不当作新增故障。没有执行训练、批量实验、交互绘图或清理主体，没有跑本批pytest。

gurobi环境缺torch、pyro、torchsort、graphviz和seaborn。四个optimizer导入探针分别卡在torch或graphviz；NoC仍报原TypeError: abstract class。未安装或删除依赖。日志见报告validation.import_probes与validation.cli。

run/sweep_seed.sh、run/exp_cmd_min_core.sh、run/exp_cmd_max_tp.sh、run/aba_scalability_scan.sh、run/abla_scalablility.sh、ablation_main.py不在tracked/approved清单。不声称它们在磁盘不存在，也不查未跟踪替代文件。旧脚本的实验可运行性未验证。

run/clean.sh的三条递归rm会处理cache/trace/log，本轮只做语法检查。共享/tmp/fd1及后台任务的旧扫描脚本也未执行。不为验证清理而清理用户数据。

## 交接

177个代码文件与B24已验收快照一致；两账本、保护文档、HEAD/index未改。报告用B24 JSON文件SHA和代码映射SHA固定这次只读基准。只有本审查包、B25报告、DFS current、双全局记录及协议会新增或更新。

请reviewer独立核对范围、文档API、环境探针、16项bash -n、6项help、28文件分类及177份SHA。按B22分析性请求的做法裁定是否KEEP_AS_IS；没有源码动作待实施。B11-B21+B23+B24共13个源码批次仍未提交，B25不计入源码批数。全仓审查仍未完成，后续沿既有DFS检查剩余独立模块。


## 最终裁决

E0102已由真实reviewer写入，28文件KEEP_AS_IS。独立重放结果存/tmp/b25_review_final/review.json，并持久复制到B25报告reviewer_acceptance：6help、16bash -n、5原导入错误及177代码/双账本/6文档哈希均吻合。不把语法/help当实验验证。

补正：NUL分隔Git清单证实含空格的run/aba_scalability_scan copy.sh是单一文件，16份shell全存在；两个假失败是reviewer命令分词问题，不是Git索引缺失。原E0102不改写，以补充事实为准。

等待期间只读的fit.py、throughput_cnt.py和test_alloc_lat.py另记报告readonly_followup，未做源码动作；test_alloc_lat会把原worktree插入sys.path，故未执行。用户现要求为积累的已验收改动建立commit快照，正在精确核对暂存清单，尚未提交或推送。
