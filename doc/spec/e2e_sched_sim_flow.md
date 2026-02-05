现在要将benchmark的生成和现有的测试串联起来：
1. 需要根据参数，生成physical graph
2. 根据调度器类型，例如，cyc，resev，pglb，需要调用相应的算法coalescing，repack以及bin_split 生成相应的应的调度参数
这些结果传递给，后面的approach中的各种case 

在尽量少改动底层代码的情况下，请对1,2 设计的代码重新包装：

1. 保证相同的文件命名规则，
2. 支持根据参数生成workload，和调度信息，也兼容手动加载
3.参照acc_p_factory 根据调度类型，绑定调度参数生成器

先给出整体思路。

run/2/aba_scalability_scan.sh run/2/abla_scalability.sh 是针对老代码实现的消融实验脚本；
main_approach.py 则目的在于串联实验步骤：workload生成，生成调度信息（step1-3），进行随机测试（step4）。

请你：
1. 检查当前main_approach中的流程和文档的描述，以及collector之间是否匹配
2. 用python实现新的脚本，针对三个消融实验：进行参数扫描，调用main_approach.py中的标准流程，最后输出统计信息，缓存相应的Statisticcollector，并完成绘图。

main_approach.py 旨在实现，端到端的调度和仿真流程：
1. workload 生成：
    - 根据参数，生成physical graph

2. 根据配置参数，生成调度信息
    - step 0（generate_workload_and_criticality）：
        - 生成workload，物理graph
    - step 1：
        - 计算时间片初始分派和资源需求，由deduce_cfg2实现，文档在doc/chain_slack_assignment_algorithm
        - 并确定资源总量，由determine_resource_config实现：由args.num_cores决定，若未指定，则根据bin_list确定。
        - 所需的资源上限，定义为，cyc case (num_bins==-1), exec_t_comp_ratioA==0.99 时的资源需求。

    - step 2：bin_split （可选）
    - step 3：repack （可选）
    其中 bin_split 和 repack 封装在perform_bin_packing中，根据num_bins,以及是否need_repack决定是否执行step2和step3；
    - num_bins == 1: 
        - 跳过step2
        - 否则执行step2
    - need_repack == True：
        - 执行step3
        - 否则跳过step3
    
3. 进行随机测试：
    - 根据调度信息，进行随机测试
    - 收集统计信息
    - 绘图

4. 消融试验：
    实验设计主要思想在于将静态调度分解为两种机制：1. 时间维度上的预留机制 2. 空间维度上的隔离机制
    预留机制对应于，时间维度的分配，由时间片调整repack（step3）实现，参数exec_t_comp_ratioB；特别的，默认所有case有一个时间片初始分派（step1），比例为exec_t_comp_ratioA；
    隔离机制对应于，空间维度的分配，在bin_split（step2），参数为num_bins；默认初始状态为一个分区。
    |case | 预留 | 隔离 | 动态 time sharing| 动态 space sharing|
    |-----|------|------|------|------|
    |cyc | √ | √| × | × |
    |glb|× | × | √ | √ |
    |reserv | √ | √ | √ | √ |
    |pglb | × | √ | √ | √ |
    |cyc-S | x | √ | √ | × |


    有如下机制组合：
    - 所有case 都需执行step0和step1，并确定资源总量，以及初始时间片分派
    - Baseline:
        - cyc: 设置num_bins = -1，额外执行step2，获取分组信息 (step0-1-2) 
        - glb: 无需额外调度信息 (step0-1)
    - reserv: 同cyc，但是需要额外执行step3，进行repack (step0-1-2-1-3)
    - 消融实验：
        - cyc-S: 设置num_bins = -1，额外执行step2，获取分组信息 (step0-1-2) 
        - pglb: 设置num_bins > 1，额外执行step2，获取分组信息 (step0-1-2) 

当前流程存在的问题：
    - 我们将所需的资源上限定义为，cyc case (num_bins==-1), exec_t_comp_ratioA==0.99 时的资源需求。
    - 同时，如果指定了 num_cores，或者指定了bin_list，则需要根据这些参数，重新计算资源需求。
    - 如果调整了分箱方案，也会影响这个数值。
    导致逻辑非常混乱，具体而言，
    - 装箱之前determine_resource_config里面封装了一个apply_forced_num_cores，根据模式，case类型以及是否强制指定核心数，是否需要repack，更新num_cores和bin_list
    - perform_bin_packing，里面封装了一个coleasing_alloc_1bin，用来确定glb的最大资源需求
    - 如果指定num_cores，分箱之后（step 2）后需要apply_forced_num_cores
    - exec_t_comp_ratioA 的含义在代码中十分的模糊

问题的关键不在于此：
    我需要你干的事，是优化流程。消融试验我已经设置好了，只是里面的参数设置，我还比较困惑。

在优化流程之前，需要现在拆解现有流程的复杂判断：
    首先，在setup_benchmark 中根据exec_t_comp_ratioA 和 exec_t_comp_ratioB 决定是否需要repack。


    应该是之前的版本，使用"sim_main.py"同时支持配置生成和仿真，test_case来选择是生成配置还是进行仿真，但是后面添加了event-driven 
     版本后端"/home/zhangchg/git_repo/scheduler/approach_sim.py" 所以前面仿真后端就用不到了，所以只会设置为bin_pack_new
    因此determine_resource_config的逻辑应该为：
    对于repack 而言，从bin_list计算资源
        如果指定了args.num_cores，则需要apply_forced_num_cores
        否则，从bin_list计算资源
    对于non-repack stage，则只需要初始化：
    - num_cores, bin_list = args.num_cores, []

    1. 非repack stage
        - determine_resource_config 只充当初始化角色
        - 在perform_bin_packing 同样走非repack stage的流程
        - 经过装箱之后，确认资源数量和分箱方案
        - 如果指定了args.num_cores，则需要apply_forced_num_cores （需要确认是否存在这种case，以及这个逻辑是否可以从装箱的逻辑中解耦出去）
            
    2. repack stage
        - 本质上，只会在每个分箱内部调整不改动分箱。是非repack stage的下一个阶段
        - determine_resource_config 这里对资源进行了二次确认，保证分箱资源的正确性。
            - 如果指定了args.num_cores，则需要apply_forced_num_cores，否则，从bin_list计算资源
        - 在perform_bin_packing 中，走repack stage的流程

可以看到，现在的代码判断逻辑层次出了问题。
1. repack 应该是最外侧的逻辑，但是现在出现在了函数内侧。
2. 非repack stage 和 repack stage 的判断逻辑出现了交织。强制资源数量这个行为，不需要进行两次。
3. 存在一些老旧的逻辑在里面，例如:
    - "args.test_case in two_stage_case_coll"
4. perform_bin_packing中封装了 太多不是很相关的逻辑，用条件判断：
    - 初始global coaleacing
    - bin split
    - repack

应该优化函数划分，优化条件判断，优化代码结构。

<!-- 
当前流程支持情况：
- perform_bin_packing 可以根据分箱数量，自动选择是否执行step2

当前流程无法支持的情况：
当前流程中 step0-1 是所有方法的前置步骤。但是这一步包装在build_simulation_env 中。
- glb 只需要执行deduce_cfg2，问题：
    - 问题：当前流程无法选择跳过step 2
- 当需要repack的时候，需要必须设置exec_t_comp_ratioB=None 执行step1-2，获取分组和分区大小；然后设置exec_t_comp_ratioB 重新执行deduce_cfg2 获取新的时间片初始分派，才能用push_task_into_bins_new进行repack，问题：
    - 当前流程，无法支持这step0-1-2-1-3的流程。
    - 无法将step2的结果分组和分区大小传递给step3 -->

