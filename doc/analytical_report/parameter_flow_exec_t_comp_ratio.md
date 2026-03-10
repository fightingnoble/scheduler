# exec_t_comp_ratioA/B 参数传递路径完整说明

## 概述

`exec_t_comp_ratioA` 和 `exec_t_comp_ratioB` 是实验中的两个关键参数，用于控制任务执行时间的分位数预留：
- **exec_t_comp_ratioA**: 初始时间片分配使用的分位数（保守）
- **exec_t_comp_ratioB**: repack 使用的分位数（激进），用于实现时间维度共享

## 完整传递路径

### 第1层：实验脚本入口

**文件**: `scripts/motiv_exp_runner.py` 或 `scripts/abla_exp_runner.py`

```python
# scripts/motiv_exp_runner.py (第38-41行)
mapping_args = {
    'exec_t_comp_ratioA': 0.99,  # 默认值
    'exec_t_comp_ratioB': -1,     # 默认不 repack
    ...
}
```

**使用示例**:
```bash
# 设置 A=0.7, B=0.5
python scripts/abla_exp_runner.py --case 1 --base_ratioA 0.7 --base_ratioB 0.5
```

---

### 第2层：参数模板构造

**文件**: `scripts/motiv_exp_runner.py` (第62-97行)

```python
class ParamTemplate:
    """参数模板，封装三类参数并提供克隆、增量更新与合并导出能力。"""
    
    def __init__(self, mapping, runtime, specific):
        self.mapping = mapping
        self.runtime = runtime
        self.specific = specific
    
    def with_updates(self, mapping=None, runtime=None, specific=None):
        # 更新参数
        new_mapping = copy.deepcopy(self.mapping)
        if mapping:
            new_mapping.update(mapping)
        return ParamTemplate(new_mapping, new_runtime, new_specific)
    
    def to_run_args(self):
        # 合并并过滤 None
        merged = {}
        for group in (self.mapping, self.runtime, self.specific):
            for k, v in group.items():
                if v is not None:
                    merged[k] = v
        return merged
```

---

### 第3层：main_approach.py

**文件**: `main_approach.py` (第14-32行)

```python
def main():
    args = input_parser()  # 解析命令行参数
    
    # 设置切换开销控制（用于Case 3对照实验）
    realloc_disabled = getattr(args, 'barrier_dis', False)
    set_realloc_disabled(realloc_disabled)
    
    # Setup benchmark: 生成/加载调度方案 (bin_list)
    args.test_case = case_name_bp
    n_p_bk, args.n_p = args.n_p, 3
    G, pid2name, bin_list, policy, path_ctx, T_hp = setup_benchmark(args, time_norm_factor)
    
    args.n_p = n_p_bk
    
    # ... 后续仿真代码
    return stats_collector
```

---

### 第4层：approach_setup.py - setup_benchmark()

**文件**: `approach_setup.py` (第58-102行)

这是**参数分发的关键点**！

```python
def setup_benchmark(args, time_norm_factor):
    print_title("Benchmark Setup Started")
    
    # 1. 构建路径和上下文
    path_params, path_ctx = build_paths_and_ctx(args)
    
    # ==================== 关键逻辑 ====================
    # 检查是否需要重新装箱（repack 模式）
    # 条件：exec_t_comp_ratioB 不为 -1 且 exec_t_comp_ratioA > exec_t_comp_ratioB
    if (args.exec_t_comp_ratioB != -1 and args.exec_t_comp_ratioA > args.exec_t_comp_ratioB):
        need_repack = True
    # ================================================
    
    # step 0-1-2: 使用 exec_t_comp_ratioA
    # 第一次调用：生成初始分配（所有方法都需要）
    args.quantile = args.exec_t_comp_ratioA
    hyper_p, bin_list = run_benchmark_setup_pipeline(
        args, path_ctx, path_params, False, None, None
    )
    
    if need_repack:
        # step 0-3: 使用 exec_t_comp_ratioB
        # 第二次调用：进行 repack（仅 reserv 方法需要）
        args.quantile = args.exec_t_comp_ratioB
        hyper_p, bin_list = run_benchmark_setup_pipeline(
            args, path_ctx, path_params, True, hyper_p, bin_list
        )
    
    # ... 后续代码
    return G, pid2name, bin_list, args.policy, path_ctx, hyper_p
```

**关键点**:
- 通过 `args.quantile` 作为中间变量转发参数
- 根据是否需要 repack，设置不同的 quantile 值

---

### 第5层：sim_main.py - perform_bin_packing()

**文件**: `sim_main.py` (第385-477行)

```python
def perform_bin_packing(args, glb_p_list, num_cores, bin_list, hyper_p,
                       sim_step, path_para_dict, para_scan_group1,
                       event_iter_dict, quantumSize, num_periods,
                       cfg_para_dict, physical_graph_nx, need_repack,
                       plot_path_para, path_ctx: PathContext, 
                       scheduler_list, monitor_list,
                       msg_dispatcher,
                       a_data_pipe, w_data_pipe
                       ):
    """执行 bin-packing 算法，生成调度表并保存"""
    
    # 设置所有任务为 hard deadline
    for _p in glb_p_list:
        _p.task.criticality = "hard"
        _p.task.chain_criticality = "hard"
    
    # 根据 binpack_cfg 的算法类型选择分支
    if args.binpack_cfg["algorithm"] == "scratch":
        # scratch 模式（不使用）
        bin_list = push_task_into_bins_new(
            bin_list, glb_p_list, ...,
            sim_step, hyper_p, args.exec_t_comp_ratioB,  # 直接使用 B
            ...
        )
    
    elif args.binpack_cfg["algorithm"] == "guided":
        from sched.global_sched import coleasing_alloc_cluster
        
        if not need_repack:
            # ==================== 第一次：Bin Split ====================
            # 使用 exec_t_comp_ratioA 进行空间分区
            split_ratio = args.exec_t_comp_ratioA  # 【第423行】
            print("="* 20 + "Bin-split mode: Cluster-based allocation" + "="* 20 + "\n")
            
            max_core_num, pid2_bin_id, bin_size_list = coleasing_alloc_cluster(
                bin_list, glb_p_list, affinity_cfg, event_iter_dict,
                None, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, split_ratio,  # 【第429行】
                ...
                n_partition = args.num_bins if args.num_bins != -1 else 9999,
                ...
            )
            # ... 更新 num_cores
        else:
            # ==================== 第二次：Repack ====================
            # 使用 exec_t_comp_ratioB 进行时间区间重新分配
            assert len(bin_list) > 0
            
            print("="* 20 + "Repack mode: Redistribute slack and rebin" + "="* 20 + "\n")
            
            # 构造局部 binpack_cfg
            binpack_cfg_local = dict(args.binpack_cfg)
            binpack_cfg_local["mapping"] = pid2_bin_id
            binpack_cfg_local["bin_sel_mod"] = "pre_defined"
            binpack_cfg_local["affinity_en"] = False
            binpack_cfg_local["affinity_level"] = 0
            
            # 使用 push_task_into_bins_new 进行重新装箱
            bin_list = push_task_into_bins_new(
                bin_list, glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.exec_t_comp_ratioB,  # 【第468行】
                ...
                binpack_cfg=binpack_cfg_local,
                ...
            )
```

**关键点**:
- `not need_repack` 时，使用 `split_ratio = args.exec_t_comp_ratioA`
- `need_repack` 时，使用 `args.exec_t_comp_ratioB`

---

### 第6层：sched/global_sched.py - push_task_into_bins_new()

**文件**: `sched/global_sched.py` (第47-160行)

```python
def push_task_into_bins_new(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, quantile,  # 【第52行】接收 quantile 参数
        job_graph: DiGraph,
        ...
        ):
    """实现一个 2D bin-packing 算法"""
    
    # ... 初始化代码 ...
    
    # ==================== 关键：将 quantile 存储到 binpack_cfg 中 ====================
    var_dist_map = {p.task.name: job_graph.nodes[p.task.name]['var_dist'] for p in glb_p_list}
    binpack_cfg = dict(binpack_cfg)  # shallow copy
    binpack_cfg['var_dist_map'] = var_dist_map
    binpack_cfg['quantile'] = quantile  # 【第121行】
    # =========================================================================
    
    # ... bin-packing 循环 ...
    
    for n_slot in range(sim_slot_num):
        # 调用 push_step_new()
        push_step_new(
            sched, msg_dispatcher, a_data_pipe, w_data_pipe, 
            n_slot, timestep, quantile,  # 【第138行】传递 quantile（但函数内部不直接使用）
            event_range, sim_slot_num, curr_t, 
            ...
            binpack_cfg=binpack_cfg,  # 【关键】通过 binpack_cfg 传递 quantile
            ...
        )
    
    # ... 返回代码 ...
```

**重要发现**:
- 第121行：将 `quantile` 存入 `binpack_cfg['quantile']`
- 第138行：虽然传递了 `quantile` 参数给 `push_step_new`，但该参数在函数体中并未被直接使用
- 真正的参数传递是通过 `binpack_cfg` 字典

---

### 第7层：sched/global_sched.py - push_step_new()

**文件**: `sched/global_sched.py` (第162-286行)

```python
def push_step_new(
        sched: Scheduler, msg_dispatcher: MsgDispatcher,
        a_data_pipe: DataPipe, w_data_pipe: DataPipe,
        n_slot: int, timestep: float, percentile: float,  # 【第162行】接收参数
        event_range: float, sim_slot_num: int, curr_t: float,
        ...
        binpack_cfg:Dict,  # 【第170行】接收 binpack_cfg
        ...
        ):
    """每个时间步的 bin-packing 推进"""
    
    # percentile 就是 quantile，但在此函数中并未直接使用
    # ... bin-packing 逻辑 ...
    
    # 根据任务优先级和资源需求进行分配
    # 使用 binpack_cfg 中的 quantile 进行资源估算
    
    # ==================== 关键：传递 binpack_cfg 给资源分配器 ====================
    if trigger_condA or trigger_condB:
        glb_alloc_new2(
            process_dict, quantumSize, timestep, 
            ready_queue, running_queue, rsc_recoder, 
            rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, 
            bin_list, bin_name_list, n_slot, curr_t, 
            binpack_cfg,  # 【第256行】传递包含 quantile 的 binpack_cfg
            show_warnings, 
            verbose, DEBUG_FG
        )
    # =====================================================================
```

**重要说明**:
- 第162行的 `percentile` 参数在函数体中**没有被直接使用**
- 真正的参数传递是通过第170行的 `binpack_cfg` 参数
- 第256行：将 `binpack_cfg` 传递给 `glb_alloc_new2`

---

### 第8层：sched/pre_alloc_new.py - glb_alloc_new2()

**文件**: `sched/pre_alloc_new.py` (第21-130行)

```python
def glb_alloc_new2(process_dict, quantumSize, timestep, 
                    ready_queue, running_queue, rsc_recoder, 
                    rsc_recoder_his:Dict[int, LRUCache], issue_list, preempt_list, iter_next_bin_obj, 
                    bin_list:TaskQueue, bin_name_list, n_slot, curr_t, 
                    binpack_cfg:Dict,  # 【第25行】接收 binpack_cfg
                    show_warnings=True, 
                    verbose:bool=False, DEBUG_FG:bool=False,
                  ):
    """将就绪任务推入空闲时间槽"""
    
    # ... 初始化和排序代码 ...
    
    while len(sorted_ready_queue):
        _p = sorted_ready_queue.queue.pop(0)
        
        # ==================== 关键：调用资源分配函数 ====================
        state = allocate_rsc_4_process_new2(
            # request parameters
            _p, n_slot,
            # sched components
            process_dict, rsc_recoder, rsc_recoder_his, 
            iter_next_bin_obj, bin_list, bin_name_list,
            # sched parameters
            timestep, quantumSize,
            binpack_cfg,  # 【第88行】传递 binpack_cfg
            preempt_list,
            show_warnings=show_warnings,
            verbose=verbose, DEBUG=DEBUG_FG,
        )
        # ============================================================
```

---

### 第9层：sched/pre_alloc_new.py - allocate_rsc_4_process_new2()

**文件**: `sched/pre_alloc_new.py` (第131-...行)

```python
def allocate_rsc_4_process_new2(
        # request parameters
        _p:ProcessInt, n_slot:int, 
        # sched components
        process_dict:Dict[int, ProcessInt], rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
        iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], 
        
        # sched parameters
        timestep, quantumSize, 
        binpack_cfg:Dict,  # 【第140行】接收 binpack_cfg
        preemption_list:List[ProcessInt],
        show_warnings=True, 
        verbose:bool=False, DEBUG:bool=False,
        ):
    """根据分位数计算每个任务的资源需求"""
    
    # Step1: 初始化资源请求参数
    time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep)
    
    # ... 其他初始化 ...
    
    # ==================== 关键：调用任务资源估算函数 ====================
    req_rsc_size, got_latency, got_constr = _p.rsc_req_estm_quantile(
        _p, slack, FLOPS_PER_CORE, binpack_cfg, constr,  # 【第162行】
        time_slot_s=None, time_slot_e=None, max_size=tot_cores
    )
    # ================================================================
    
    # ... 返回资源分配结果 ...
```

---

### 第10层：task/task_agent.py - rsc_req_estm_quantile()

**文件**: `task/task_agent.py` (第563-592行)

```python
def rsc_req_estm_quantile(
    _p, slack, FLOPS_PER_CORE, binpack_cfg, constr:TaskConstraints,
    max_size=float("inf")
    ):
    """基于分位数的核心大小计算（使用 var_dist）"""
    
    # ==================== 关键：从 binpack_cfg 获取 quantile ====================
    var_dist_map: Dict[str, Variation] = binpack_cfg.get('var_dist_map', None)
    q = binpack_cfg.get('quantile', None)  # 【第569行】
    # =================================================================
    
    node_name = _p.task.name
    dist = var_dist_map[node_name]
    
    if isinstance(dist, SenVarDist):
        # Sensor-like task: 固定延迟分位数，core=1
        req_rsc_size = 1
    else:
        # Acc task: load_q/(cores*FLOPS_PER_CORE) + io_q <= window_time
        try:
            # ==================== 关键：使用 quantile 进行分位数计算 ====================
            load_q = float(dist.load_dist.quantile(q))  # 【第578行】负载分位数
            io_q = float(dist.exec_dist.quantile(q))   # 【第579行】执行延迟分位数
            # ============================================================
        except Exception:
            # Fallback: 将其视为零IO的纯计算任务
            load_q = float(getattr(dist, 'quantile', lambda qq: _p.remburst)(q))
            io_q = 0.0
        
        compute_budget = slack - io_q
        if compute_budget <= 0:
            # 没有计算时间；请求最大允许核心数
            req_rsc_size = max(1, getattr(_p, 'core_max', 1))
        else:
            # 根据负载和计算预算计算理想核心数
            ideal_cores = int(math.ceil(load_q / (compute_budget * FLOPS_PER_CORE)))
            req_rsc_size, got_constr = find_legal(constr, max_size, ideal_cores)
            got_latency = elim_nume_error(_p.task.flops / req_rsc_size / FLOPS_PER_CORE)
    
    return req_rsc_size, got_latency, got_constr
```

**关键说明**:
- 第569行：从 `binpack_cfg['quantile']` 获取分位数
- 第578行：使用 `q` 计算负载分位数 `load_q`
- 第579行：使用 `q` 计算执行延迟分位数 `io_q`
- 这两个分位数决定了任务的资源需求 `req_rsc_size`

---

## 完整流程图

```
【外层】实验脚本
    ↓ 设置参数
exec_t_comp_ratioA=0.99, exec_t_comp_ratioB=-1
    ↓
【第2层】ParamTemplate
    ↓ 封装参数
to_run_args() → merged dict
    ↓
【第3层】main_approach.py
    ↓ 调用
setup_benchmark(args)
    ↓
【第4层】setup_benchmark() (approach_setup.py)
    ├── 检查: if (exec_t_comp_ratioB != -1 and exec_t_comp_ratioA > exec_t_comp_ratioB)
    ├── 第1次: args.quantile = exec_t_comp_ratioA
    │   └── run_benchmark_setup_pipeline(need_repack=False)
    └── 第2次: if need_repack:
            └── args.quantile = exec_t_comp_ratioB
                └── run_benchmark_setup_pipeline(need_repack=True)
                    ↓
【第5层】perform_bin_packing() (sim_main.py)
    ├── not need_repack:
    │   └── split_ratio = exec_t_comp_ratioA (第423行)
    │       └── coleasing_alloc_cluster(split_ratio) (第429行)
    │           └── quantile = exec_t_comp_ratioA (第645行，通过参数传递)
    │               └── coleasing_alloc_1bin(..., quantile)
    └── need_repack:
        └── args.exec_t_comp_ratioB (第468行)
            └── push_task_into_bins_new(quantile)
                └── binpack_cfg['quantile'] = quantile (第121行)
                    ↓
【第6-7层】push_step_new() 和 glb_alloc_new2()
    ├── push_step_new(percentile) ← 参数未使用
    └── binpack_cfg ← 包含 'quantile' 字段
        └── glb_alloc_new2(binpack_cfg)
            └── allocate_rsc_4_process_new2(binpack_cfg)
                ↓
【第8-10层】allocate_rsc_4_process_new2() 和 rsc_req_estm_quantile()
    └── _p.rsc_req_estm_quantile(_p, ..., binpack_cfg)
        └── q = binpack_cfg.get('quantile', None) (第569行)
            └── load_q = dist.load_dist.quantile(q) (第578行)
            └── io_q = dist.exec_dist.quantile(q) (第579行)
                └── 计算资源需求 req_rsc_size
```

---

## 关键验证点

### 1. approach_setup.py 中的逻辑

```python
# approach_setup.py (第78-92行)
if (args.exec_t_comp_ratioB != -1 and args.exec_t_comp_ratioA > args.exec_t_comp_ratioB):
    need_repack = True

args.quantile = args.exec_t_comp_ratioA  # 第一次
hyper_p, bin_list = run_benchmark_setup_pipeline(..., False, None, None)

if need_repack:
    args.quantile = args.exec_t_comp_ratioB  # 第二次
    hyper_p, bin_list = run_benchmark_setup_pipeline(..., True, hyper_p, bin_list)
```

### 2. sim_main.py 中的使用

```python
# sim_main.py (第419-477行)
if not need_repack:
    split_ratio = args.exec_t_comp_ratioA  # 第423行
    coleasing_alloc_cluster(..., split_ratio)  # 第429行
else:
    push_task_into_bins_new(..., args.exec_t_comp_ratioB)  # 第468行
```

### 3. sched/global_sched.py 中的传递

```python
# push_task_into_bins_new (第52, 121, 138行)
def push_task_into_bins_new(..., quantile, ...):
    binpack_cfg = dict(binpack_cfg)
    binpack_cfg['quantile'] = quantile  # 第121行：存储到 binpack_cfg
    push_step_new(
        sched, msg_dispatcher, a_data_pipe, w_data_pipe, 
        n_slot, timestep, quantile,  # 第138行：传递（但未使用）
        event_range, sim_slot_num, curr_t, 
        ...
        binpack_cfg=binpack_cfg,  # 通过 binpack_cfg 传递
        ...
    )
```

### 4. task/task_agent.py 中的使用

```python
# rsc_req_estm_quantile (第569, 578-579行)
def rsc_req_estm_quantile(_p, slack, FLOPS_PER_CORE, binpack_cfg, constr):
    q = binpack_cfg.get('quantile', None)  # 第569行：获取 quantile
    load_q = float(dist.load_dist.quantile(q))  # 第578行：负载分位数
    io_q = float(dist.exec_dist.quantile(q))   # 第579行：执行延迟分位数
    
    compute_budget = slack - io_q
    ideal_cores = int(math.ceil(load_q / (compute_budget * FLOPS_PER_CORE)))
    req_rsc_size, got_constr = find_legal(constr, max_size, ideal_cores)
    
    return req_rsc_size, got_latency, got_constr
```

---

## 不同方法的使用模式

### cyc (纯静态周期调度)

```python
exec_t_comp_ratioA = 0.5/0.6/0.7/0.8/0.9/0.99  # 扫描
exec_t_comp_ratioB = -1  # 不 repack
need_repack = False

# 实际调用：
# 第1次: quantile = exec_t_comp_ratioA → bin_split (空间分区)
# 第2次: 不调用
```

### glb (纯动态全局调度)

```python
exec_t_comp_ratioA = 0.99  # 高分位数
exec_t_comp_ratioB = -1  # 不 repack
num_bins = 1  # 跳过聚类

# 实际调用：
# 第1次: quantile = exec_t_comp_ratioA → bin_split (但 n_partition=1 跳过)
# 第2次: 不调用
```

### pglb (分区全局动态)

```python
exec_t_comp_ratioA = 0.7  # 中等分位数
exec_t_comp_ratioB = -1  # 不 repack
num_bins = 1/2/4/8  # 扫描

# 实际调用：
# 第1次: quantile = exec_t_comp_ratioA → bin_split (空间分区)
# 第2次: 不调用
```

### reserv (预留调度 - 混合方法)

```python
exec_t_comp_ratioA = 0.7  # 中等分位数
exec_t_comp_ratioB = 0.5  # 低分位数（更激进）
num_bins = 4

# 实际调用：
# 第1次: quantile = exec_t_comp_ratioA → bin_split (空间分区)
#       └── coleasing_alloc_cluster(split_ratio=0.7)
#           └── binpack_cfg['quantile'] = 0.7 (从参数传递)
# 第2次: quantile = exec_t_comp_ratioB → repack (时间区间重新分配)
#       └── push_task_into_bins_new(quantile=0.5)
#           └── binpack_cfg['quantile'] = 0.5
```

### cyc(S) (空间分区 + 时间共享)

```python
exec_t_comp_ratioA = 0.7  # 中等分位数
exec_t_comp_ratioB = 0.5/0.6/0.7/0.8/0.9  # 扫描
num_bins = -1  # 不分箱

# 实际调用：
# 第1次: quantile = exec_t_comp_ratioA → bin_split (但 num_bins=-1 跳过)
# 第2次: quantile = exec_t_comp_ratioB → repack (时间区间重新分配)
#       └── push_task_into_bins_new(quantile=0.5)
#           └── binpack_cfg['quantile'] = 0.5
```

---

## 总结

### 传递路径总结

1. **参数入口**: `scripts/motiv_exp_runner.py` 或 `scripts/abla_exp_runner.py`
2. **中间转发**: `args.quantile` 在 `setup_benchmark()` 中作为转发变量
3. **两次调用**:
   - 第1次: 使用 `exec_t_comp_ratioA` → `coleasing_alloc_cluster(split_ratio)`
     - 参数直接传递：`coleasing_alloc_cluster(..., quantile=exec_t_comp_ratioA)`
   - 第2次: 如果 `need_repack`，使用 `exec_t_comp_ratioB` → `push_task_into_bins_new(quantile)`
     - 参数传递：`push_task_into_bins_new(..., quantile=exec_t_comp_ratioB)`
4. **binpack_cfg 中转**: 
   - `push_task_into_bins_new` 将 `quantile` 存入 `binpack_cfg['quantile']`
   - `push_step_new` 的 `percentile` 参数**未被使用**，实际通过 `binpack_cfg` 传递
5. **最终使用**: `rsc_req_estm_quantile()` 从 `binpack_cfg['quantile']` 读取分位数
   - 用于计算：`load_q = dist.load_dist.quantile(q)`
   - 用于计算：`io_q = dist.exec_dist.quantile(q)`

### 关键文件和代码位置

| 文件 | 行号 | 关键代码 |
|------|------|----------|
| approach_setup.py | 78-92 | 参数转发逻辑（args.quantile） |
| sim_main.py | 423, 468 | split_ratio 和 exec_t_comp_ratioB |
| sched/global_sched.py | 52, 121 | push_task_into_bins_new 参数和 binpack_cfg 存储 |
| sched/global_sched.py | 162-286 | push_step_new（percentile 未使用，通过 binpack_cfg 传递） |
| sched/pre_alloc_new.py | 25, 88 | glb_alloc_new2 和 allocate_rsc_4_process_new2 |
| task/task_agent.py | 563-592 | rsc_req_estm_quantile（最终使用分位数） |
| task/task_agent.py | 569, 578-579 | 从 binpack_cfg 获取 quantile 并使用 |

### 重要发现

**push_step_new 函数中的 `percentile` 参数是一个"僵尸参数"**：
- 在函数签名中接收 `percentile: float` 参数
- 但在函数体中完全没有使用这个参数
- 真正的参数传递是通过 `binpack_cfg` 字典
- 这是一个代码遗留或设计不一致的问题

实际的参数传递路径：
```
quantile → binpack_cfg['quantile'] → glb_alloc_new2(binpack_cfg) 
    → allocate_rsc_4_process_new2(binpack_cfg) 
    → rsc_req_estm_quantile(binpack_cfg)
    → q = binpack_cfg.get('quantile')
    → load_q, io_q = dist.quantile(q)
```
