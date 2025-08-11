
## 20250703
完善physical 和 logical graph的构建流程：保证后续能够根据概率图获取调度，验证所需的所有信息

New features:
 使用简单的调度器，验证调度器是否能够正确调度

## 20250726
    统一acc_p和sen_p的接口，方便后续的扩展
    
## 20250806 - 0811
1. 扩展多个超周期下的case的生成：
    需要循环多个超周期
2. 每次生成负载的时候，通过seed生成可控的，exp_comp_t
    每个超周期下，任务的offset需要加上当前超周期的offset，进而影响到ert和ddl，以及静态调度表。
    参考scheduler_base 中的参数设置方法，
    为Mygraph 添加成员函数，输入为指示当前第几个超周期的hp_idx和一个种子seed，
    将节点复制一份，节点名称后面添加下划线+hp_idx，
    同时将这些信息添加到，n_pred_map, ..., ert_map 这些状态缓存中，
    除了sink_node之外的src和op予随机的执行时间。
    

