# 深度优先探索：当前进度

## 范围和限制

B11、B12已经真实reviewer验收，均未提交。按用户持续目标开始全项目扫描，不重启取消或暂缓的Repack请求。

初始静态快照（B13前）包含137个Python文件，全部AST解析成功；三个入口的保守模块依赖得到72个文件，索引519个顶层函数/类。127个tracked Python路径中，B12两个旧根路径已移走；加入当时12个明确批准新源文件后共137个。下表保留该时点分组，不冒充实时清单。B13增加归档/测试两文件、B14增加测试一文件、B15增加归档/测试两文件，当前142个Python已全部AST解析，增量证据在各批JSON。16个shell脚本尚未全量语义审查。

| 区域 | Python文件数 |
| --- | --- |
| 根目录 | 33 |
| sched | 33 |
| model | 18 |
| analyze | 12 |
| approach | 8 |
| task | 8 |
| scripts | 7 |
| optimizer | 5 |
| example | 4 |
| old | 4 |
| mapper / run / cleanup | 2 / 2 / 1 |

静态覆盖不等于每个模块都完成了人工审查。动态导入、反射、外部运行方式和文档约定仍可能产生静态图之外的使用；没有根据扫描结果自动生成删除授权，既有测试和类内方法保持保护。

## 当前深入的路径

main_approach -> approach_sched根兼容入口 -> approach/approach_sched.py -> utils.py。utils已阅读全文，并继续核对其导入、公共配置和现存调用方。旧approach示例的另一条路径也通向utils.core_distr，B12已证明这个共享工具必须保留。

当前分组：

- 活数值/分配工具：core_distr、vectorized_core_allocation，继续保留。
- 活输入、pickle、计时/分析辅助：有实际导入或调用，保留。
- 旧路径构造：build_path_old仍被sim_main.build_paths_and_ctx调用，后者被sim_main_old入口引用，保护文档也有约定。不能因为名字带old就移走。
- HDF5分块工具：get_next_chunk_id、save_chunk、load_h5_file及CHUNK_SIZE已整体分离到utils_unused.py；真实reviewer在E0045独立复测后验收。原行为保留，活utils不再加载专用h5py依赖，依赖清单不删。
- B14路径辅助：get_log_path_str、_normalize_path、check_paths_equal经E0047批准，原样追加到utils_unused，E0049已独立验收。软比较与活严格compare_paths不等价，目录getter与PathContext日志文件getter也不等价，不把归档说成已替换路径体系。
- Found保留，不因一个无引用提示开始处理类。get_cfg_n还有6处shell经run/cfg_parser.py使用；pyinstr_profiler有mapper装饰器调用，公共分析/分配/pickle工具保留。prepare_induced_env_if_needed暂REVIEW，涉及历史induced runtime，不混入B14。

## 下一步

B13/B14/B15均已独立验收，B15为E0055；E0051的CSV证据勘误保持关闭。B15新13项通过、七套联合85通过/5原有失败、三入口正常；完整导出双进程重演已核实。B11-B15均未提交，当前无在途请求，但全仓语义审查尚未完成。

global_var已完成本轮导出、内部表达式依赖、文档引用和实际星号命名空间检查：精度/路径/现用参数保持，仅十个已批名字迁出。随后返回approach_sched的实际依赖approach_Eq，已读全文；find_legal虽在Eq内未调用，仍被task_agent和chain_slack_assign经Eq导入，属于必须保留的重导出。下一条只读路径是sched/ref_alloc_search.py；类内方法不处理，未完成语义审查的模块仍为REVIEW。本次不启动新源码批次。

机器清单：cleanup/reports/dfs-exploration-inventory-20260906.json。用户不需要逐个打开原始条目，当前动作与历史继续以CLEANUP_STATUS.md、FILE_ADJUSTMENT_RECORD.md为准。


## ref_alloc_search 与 Eq 的进一步核对

ref_alloc_search全部函数属于find_legal活链，TaskConstraints与末尾断言保留；原main已在gurobi环境跑通。Eq全读后，仅norm_inv_cdf/exp_quantile两完整函数提出B16，其他数值/时间工具、分布类及方法保持。142Python及tracked文本无两候选引用，21个显式Eq导入站点不引它们；两个函数的18组值/异常与45公共名已固定，不能将未用helpers称为分布类的等价替代。

B16已由E0059独立验收，当前无在途请求。新24项通过，联合109通过/5原有失败，ref_alloc原输出一致；144Python语法覆盖仍不是全仓语义审查完成。根壳未改，旧根/包两个函数名和旧函数pickle引用退休。H1哈希及H2过滤口径已完成记录澄清，无进一步源码修改。


## approach_def 顶层接口与可变状态

已读九个顶层函数及其使用路径。四个setter属于现用控制接口，set_miss_disabled虽然没有AST裸名调用，仍由B11兼容测试通过字符串参数/getattr调用，不能把静态0当成无用。get_drop_disabled在Acc_p中有实际调用，build_logical_graph由MyGraph构造调用，print_if_verbose被多个活模块使用。

get_miss_disabled/get_realloc_disabled没有直接名字使用，但原函数通过模块全局字典读取MISS_DISABLED/REALLOC_DISABLED。临时进程原样执行这两个函数到复制了布尔值的新全局字典，再调用活setter：原getter返回True，复制版仍False。仅移动函数并补from-import会破坏状态共享；保持原语义需额外机制或改写，不属于本轮授权，暂留。五个类MyGraph/GlobalEvent_t/BaseProcessor/Sen_p/Acc_p内部按用户规则不清理。

父模块approach_sched的项目导入依赖utils/Eq/approach_def已按上述路径核查。下一步回到其顶层调度函数及动态绑定，先理解使用方式，不提前判定其旧分支或类方法无用。

## runtime 绑定与 task_cfg 历史候选

approach_sched六个顶层函数全部KEEP。五策略加单分区pglb降级共6配置、8个真实Acc_p对象的绑定及空队列钩子通过；no_trigger为cyc/cyc-S的有效钩子。collector完整类与测试main保留，实际统计依赖ref_tdigest已通过真实类型、数据摘要和序列化往返核对。以上不等于完整负载下的调度正确性证明。

initiator五个函数在主流程或内部加载链中活用。old_timestep/old_stable_hp/old_T_hp仍用于调度表时间换算；old_num_hp暂REVIEW，不能连带改前三项。图加载沿task_cfg JSON入口回到Eq分布恢复；现用gen_workloads调用creat_physical_graph。

REQ-015/B17仅提案：将task_cfg的59行注释旧绘图实现与147行完整creat_jobTask_graph原样移到task_cfg_old.py。11组原行为已固定，修改前109passed/5原有failed；等待reviewer批准，尚未改源码或新建测试。其他旧allocator借用函数与现用绘图保持。task_cfg仍是部分语义审查，后续沿其依赖深入，不随机跳查。

新机器证据：cleanup/reports/dfs-approach-runtime-review-20260906.json及b17-task-cfg-legacy-graph-baseline.json。全局状态仍以CLEANUP_STATUS为准。

B17执行进度：真实E0061/E0062批准后，206行原样分离，新增18项通过；联合127passed+5原有failed、三入口通过、159非本批SHA原样。E0063已提交等待独立验收。reviewer曾误判旧allocator借用者，已通过直接文件证据自行勘误；未过滤os.walk清单不作为验收范围。

继续只读补阅现用load_taskattrib/gen_taskint_from_cfg及graph_scaling全部代码。build_node_relationship用于当前物理图展开；build_data_node_relationship还会生成显式数据节点，并被其main示例使用。两者不是等价重复，不合并；独立示例保留。下一步仍沿task_cfg深层依赖，不开启并行源码批次。

## B17 已关闭与后续路径

E0064独立验收B17，reviewer另补齐159份非本批SHA全量遍历，0缺失/不一致。Codex最终18项复测通过，所有源码哈希、账本前缀和限定恢复检查仍一致，无新事件或源码批次。

下一条路径仍是task_cfg的实际依赖。已定位task_agent七个类及load_task_from_cfg独立CLI入口，并读取TaskQueue完整实现；类内先不管，不能因外部调用少就搬走独立入口。task_cfg剩余函数与任务模型、slack_estim/load_cfg仍需继续语义审查，未宣称全仓完成。

## 模型依赖：B18 待审

本轮继续到Context_message/resource_agent/e2e_latency/performance。trace_analyser有analyze_timing实际调用，性能四接口也活；ContextMsg、资源与任务类整体保持。e2e_latency的八个完整函数已读，两未用生成器提出REQ-016/B18，其余六个有现用或历史调用者，保持不动。

naive_period_event_gen和e2e_var_sim共26行拟放同目录e2e_latency_unused.py，原样保留而不称其与离散事件版本等价。2023文档旧调用及同前缀CLI配置均保留。13组原行为与81个公共名已固定；修改前九套127passed+5原有failed。尚未改源码，E0065等待真正reviewer批准。

B18现已实施，E0071等待验收。E0067明确旧函数引用pickle是真实退休边界；E0068单列reviewer守卫变化；E0070允许还原工具多加的1字节EOF换行。最终源纯删除26行且完整SHA/EOF符合原预期；新增25通过，联合152passed+原5failed，三入口通过。

当前更新：B18已由E0072独立验收。reviewer补齐完整十套157项及新增25/25，并验证限定恢复patch；Codex直接读取其XML确认范围/计数和原5项失败一致。无在途请求，后续沿下述slack/graph依赖继续审查，不把部分覆盖说成全仓完成。

等待期间沿task_cfg→slack_estim读完其顶层函数及9个lambda，现用分配与fixcore链保留，绘图KEEP_AS_IS不变。build_score_dict_ref_flops/get_chains_info暂无代码调用证据，仅REVIEW，未另提案。graph_breakdown全文已读，真实DAG四条路径可返回；自带示例传int抛TypeError、sort助手样例抛IndexError，均作为历史行为记录，不修测试不删代码。链求解器两个顶层函数互相调用，Gurobi类暂不拆；loadA七个配置导出有task_cfg调用，保留。task_cfg.load_taskint还被自己的main调用，不能只按old allocator借用理解。
