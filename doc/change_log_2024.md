## 20240331

1. redefine the meaning of the wsc_slack_ratio: 
   
  - Old ver: the percentage of the worst-case slack relative to the e2e latency, and 
    ``` python
    wsc_slack = wsc_slack_ratio * e2e_latency,
    avg_slack = (1-exec_t_comp_ratioA) * e2e_latency,
    # the extra resource 
    RDA_size = math.ceil(size * (1-exec_t_comp_ratioA) / wsc_slack_ratio) - size 
    ```
    However, the old version does not considers the impact of other compesation factors, such as the slot grid displacement and arrival jitter. Thus we redefine the meaning of the wsc_slack_ratio as:
  - New ver: the percentage of shrinking relative to the estimated worst-case slack, which considers both the slot grid displacement and the arrival jitter besides the slowdown rate, 
  ``` python
    wsc_estm1 = (e2e_latency - jitter - temporal_abs of all nodes) * (1-slowdown ratio)
    wsc_comp = (e2e_latency - jitter_t_comp - temporal_abs of all nodes) * (1-temporal_rel)
    compact_slcak = slack_from_WC_analysis/wsc_slack_ratio 
    # the extra resource 
    RDA_size = math.ceil(size * wsc_comp / wsc_estm1 * wsc_slack_ratio) - size 
  ```
  wsc_slack_ratio > 1 
2. fix the numerical tollerance:

  - round flops to when reading from csv, and regular the upper bound of core number
    ``` python
    flops_on_path = elim_nume_error(task_attr["Flops on path (G)"]/1e3)
    core_max_compile = parallel_cfg_compile["max"] if "max" in parallel_cfg else 1000
    core_max = parallel_cfg["max"] if "max" in parallel_cfg else 1000
    ```
  - add slack to the constraint of the gurobi model, i.e. `time1n_error_tol_abs`, `flop1n_error_tol_abs`
    ``` python
        self.model.addConstr(self.flops[i] + flop1n_error_tol_abs <= self.core[i] * self.lat[i] * FLOPS_PER_CORE, name=f'flops[{i}]')
        self.model.addConstr(sum([self.lat[i] / (1 - self.margin[i][1]) + self.margin[i][0] for i in range(self.K)]) + time1n_error_tol_abs <= self.e2e, name="e2e")
    ```
## 20240403
1. add solution checking for resource estimation for average based, i.e., `check_sol` in slack_estim.py. 
  - add slack to when return the remaining slack to the tasks,
    ``` python
        slack_rem -= sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in flops_dict]) + time1n_error_tol_abs
    ```

## 20240407

1. refine the parameter setting code in repacking mode: 
    ``` python
            # assert args.binpack_cfg["algorithm"] == "bin_split"
            assert args.binpack_cfg["slack_sharing"] == False
            # 1. cheat the non-sharing model: 
            #    keep the original exec_t_comp_ratioA in slack distribution and resource estimation
            #    to make sure the repacking step use the same Bin configuration as the original one.
            # 2. backup parameters
            exec_t_comp_ratioA_bk = args.exec_t_comp_ratioA
            # 3. change the exec_t_comp_ratioA to args.exec_t_comp_ratioB in th repacking step
            args.exec_t_comp_ratioA = args.exec_t_comp_ratioB 
            args.binpack_cfg["slack_sharing"] = True
            # reset the ddl and ert
            hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args)
    ``` 

2. remove the csv recoding in "main.py" to enable parallel running 
3. Redefine the file_surffix as: 
  ```shell
  Jitter_sym="var_${jitter_comp_cfg}(J)"
  w_slowdown_sym="${Jitter_sym}_${exec_t_comp_ratioA}(T)"
  w_ld1_sym="${w_slowdown_sym}_${load_bursty_ratio1}(LD1)"
  ```
  contrast with 
  ``` python
  cfg_root_fmt = r"x{aux_scale_factor}_{e2e_latency}s_rda-{jitter_t_comp_ratio:.2%}(J)_{wsc_slack_ratio:.2%}(T)_{exec_t_comp_ratioA:.2%}(S)_ignore"
  ```

## 0727
### Slowdown modeling 
修改latency 模型：

分离模型+预分配+runtime model+runtime allocation

```latex
\(\s*1\s*- 
```
1. 分离模型
    
    新建model/performance.py：
    
    ```python
    cal_lat = lambda lat, lat_jitter=0, var_sl=1, var_ld=1: lat*(1+var_sl) * var_ld + lat_jitter
    slack_comp = lambda slack, lat_jitter=0, var_sl=1, var_ld=1: (slack - lat_jitter) / (1+var_sl) / var_ld 
    ```
    
    avg_trasfer_time 从 /model/message/data_pipe.py 移动到 model/performance.py
    

---

2. **预分配**
    
   1. Compensation:
      1. sched/packing_solver/gurobi_MP_chain_assign.py
          ```python
          # define_constraints
          # self.model.addConstr(sum([self.lat[i] / (1 - self.margin[i][1]) + self.margin[i][0] for i in range(self.K)]) + time1n_error_tol_abs <= self.e2e, name="e2e")
          self.model.addConstr(sum([cal_lat(self.lat[i], **self.margin[i]) for i in range(self.K)]) + time1n_error_tol_abs <= self.e2e, name="e2e")
          
          # check_sol
          # e2e_sum = sum([sol[i][1] / (1 - self.margin[i][1]) + self.margin[i][0] for i in range(self.K)])
          e2e_sum = sum([cal_lat(sol[i][1], **self.margin[i]) for i in range(self.K)])
          ```
    
       2. task/task_agent.py 的 rsc_req_estm
       3. slack_estim.py
    
       4. sched/global_sched.py
    
          ```python
                          # size_del_rda = process_dict[pid].task.flops/FLOPS_PER_CORE/(ddl_t-start_t)
                          # cores_dict[pid] = int(math.ceil(size_del_rda/(1-exec_t_comp_ratioB)))
                          slack = slack_comp((ddl_t-start_t), 0, exec_t_comp_ratioB)
                          size_del_rda = process_dict[pid].task.flops/FLOPS_PER_CORE/slack
                          cores_dict[pid] = int(math.ceil(size_del_rda))
          ```
          
          发现个奇怪的东西？？为啥这个地方是ratioB
---
3. **runtime model**
    
    model/resource_agent.py
    
    ```python
    # self.get_real_ops = lambda exp_ops: (1-self.var_gen()) * exp_ops
    self.get_real_ops = lambda exp_ops: slack_comp(exp_ops, 0, self.var_gen())
    ```
    

---

4. **runtime compensation**
    
    ```python
                elif chunk_s < n_slot < chunk_e:
                    # case 1: release late !!! the running task that is identified as preemptable [chunk_s, chunk_e] 
                    assert chunk_e == curr_cfg.slot_e + 1
    								# 调整前
                    # req_rsc_size = math.ceil(planned_flops/(chunk_e - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate)) 
    								# 调整后
                    slack = slack_comp((chunk_e - n_slot), 0, sched.over_provision_rate)
                    req_rsc_size = math.ceil(planned_flops/slack/timestep/FLOPS_PER_CORE) 
    						# ...
                else:
                    if round(planned_flops, flop1u_error_tol_bit) > round(chunk_flops, flop1u_error_tol_bit):
                        # case 2: previous chunk is late
                        #   newest assigned budget is still available but not enough
    										# 调整前
                        # req_rsc_size = math.ceil(planned_flops/(chunk_e + 1 - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate))
                        # 调整后，修了个bug？？
                        # req_rsc_size = math.ceil(planned_flops/(chunk_e - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate)) 
                        slack = slack_comp((chunk_e - n_slot), 0, sched.over_provision_rate)
                        req_rsc_size = math.ceil(planned_flops/slack/timestep/FLOPS_PER_CORE) 
    
    ```
    
    此处发现一个奇怪的东西，在previous chunk late的时候，chunk_e ＋1，late的case下，之前也+1 但是某一个版本被删除了

## 20240801

1. add bin split model, targeting using minimal number of bins, and do not allowing colocation of the tasks in the same bin at the same time
2. add test to evaluate the improvement with temporal sharing 
3. add batch running script: run/time_sharing.sh, and plot script: analyze/UE_extract.py

TODO: 
  test iteration optimize of ST-placement
  test bin_sharing's contribution to the unexpected conners

