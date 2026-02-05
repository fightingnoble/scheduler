from utils import elim_nume_error

from global_var import FLOPS_PER_CORE

from approach_Eq import (
    get_task_load_and_base_size,
)
from approach_def import (
    Acc_p, 
    Sen_p, 
    MyGraph, 
    GlobalEvent_t, 
    print_if_verbose,
)
from approach_sched import acc_p_factory, PartitionConfig
from approach_collector import StatisticsCollector

# TODO: These numbers are temporal magic numbers, which must be removed. 
old_timestep = 10e-6
old_num_hp = 5
old_stable_hp = 3
old_T_hp = 0.1

def get_partition_info(bin_list, graph:MyGraph, pid2name=None):
    # 1. 分区的任务映射
    partition_task_map = []
    for _bin in bin_list:
        task_set = set()
        for rsc_agent in _bin.scheduling_table:
            task_set.update(rsc_agent.rsc_map.keys())
        if pid2name is not None:
            partition_task_map.append([pid2name[pid] for pid in task_set])
        else:
            partition_task_map.append(list(task_set))
    
    
    # map the sink node to the partition
    G = graph.logical_graph
    for node in [n_ for n_ in G.nodes if G.nodes[n_]['type'] == "sink"]:
        preds = G.pred[node]
        assert len(preds) == 1, f"sink node {node} should have only one predecessor, but got {len(preds)}"
        pred = list(preds.keys())[0]
        for i, partition_tasks in enumerate(partition_task_map):
            if pred in partition_tasks:
                partition_task_map[i].append(node)
                break
        else:
            assert False, "sink node should be assigned to a partition"


    bin_name_list = [bin.name for bin in bin_list]
    # 2. 分区的大小
    partition_size = [_bin.num_resources for _bin in bin_list]

    # 3. cyclic方法下的静态调度表
    TSMap_list = []
    for _bin in bin_list:
        # record the first event of each each task in the bin
        occ_ = _bin.index_occupy_by_id()
        occ_1st = {}
        for task_id, (s, size, l) in occ_.items():
            occ_1st[task_id] = elim_nume_error(s[0]*old_timestep)
        
        TSmap = []
        left_bound = elim_nume_error((old_stable_hp-1)*old_T_hp)
        right_bound = elim_nume_error(old_stable_hp*old_T_hp)
        for cfg_slot_s, next_cfg, cfg_slot_num in getattr(_bin, "sparse_list", []):
            t_s = elim_nume_error(cfg_slot_s*old_timestep)
            t_e = elim_nume_error(cfg_slot_s*old_timestep + cfg_slot_num*old_timestep)
            if left_bound >= t_e:
                continue
            elif t_s < left_bound and t_e > left_bound:
                loc_t = elim_nume_error(0)
            elif left_bound <= t_s < right_bound:
                loc_t = elim_nume_error(t_s - left_bound)
            else:
                break
            
            assert pid2name is not None, "pid2name is required"
            cfg_ = {}
            for pid, size_ in next_cfg.items():
                if occ_1st[pid] > loc_t:
                    name_ = pid2name[pid] + "_" + str(-1)
                else:
                    name_ = pid2name[pid] + "_" + str(0)
                cfg_[name_] = size_
            TSmap.append((loc_t, cfg_))
        
        TSMap_list.append(TSmap)


    # 打印结果
    # print("Partition-Task Mapping:", partition_task_map)
    # print("Partition Size:", partition_size)
    # print("Cyclic Static Schedule:", cyclic_static_schedule)    

    return len(bin_list), partition_size, [FLOPS_PER_CORE]*len(bin_list), partition_task_map, TSMap_list 


# seting_size, time_unit, base_pwr
# if int_slot is True, the time_unit is 1, and all exe_comp_t should divide by timestep;
# otherwise, the time_unit is timestep
def load_graph_from_json(json_path, time_norm_factor):

    from task.task_cfg import load_json_graph_utils
    G = load_json_graph_utils(json_path)
    nodes = G.nodes
    edges = G.edges

    # get the pid2name map
    pid2name = {n_att['node_id']: n for n, n_att in G.nodes(data=True)}

    # 分类节点
    srcs, ops, sinks = {}, {}, {}
    task_attr, src_attr, sink_attr = {}, {}, {}

    # 先统计所有节点的入度和出度
    src_nodes = [n for n, x in G.in_degree() if x == 0]
    sink_nodes = [n for n, x in G.out_degree() if x == 0]

    # if int_slot is True, the time_unit is 1, and all exe_comp_t should divide by timestep;
    # otherwise, the time_unit is timestep

    # 分类
    for node_id in nodes:
        node = nodes[node_id]

        exp_comp_t, base_size = get_task_load_and_base_size(node, time_norm_factor)
        offset = elim_nume_error(node['offset'])
        # 源节点
        if node_id in src_nodes:
            # 找到所有后继
            srcs[node_id] = list(G.successors(node_id))
            src_attr[node_id] = {
                'offset': offset,
                'exp_comp_t': exp_comp_t, # node['comp_ratio'] / node['freq'],
                'base_size': base_size,
                'freq': node['freq'],
                'var_dist': node['var_dist'],
                # 'tgt_device': n.get('tgt_device', 'sen_p0')
            }
        # 汇节点
        elif node_id in sink_nodes:
            sinks[node_id] = []
            sink_attr[node_id] = {
                'exp_comp_t': exp_comp_t,
                'base_size': base_size,
                'offset': offset,
                'ddl': node['ddl'] + offset,
                # 'tgt_device': n.get('tgt_device', 'sink')
            }
        # 中间节点
        else:
            ops[node_id] = list(G.successors(node_id))
            task_attr[node_id] = {
                'offset': offset,
                'exp_comp_t': exp_comp_t,
                'exp_io_t': node['exp_io_t'],
                'base_size': base_size,
                # 'tgt_device': n.get('tgt_device', 'acc_p0'),
                'var_factor': node['var_factor'],
                'ert': node['ert'] + offset,
                'ddl': node['ddl'] + offset,
                'var_dist': node['var_dist'],
            }

    return srcs, ops, sinks, task_attr, src_attr, sink_attr, pid2name
    # TODO: duplicate nodes in the graph, add noise to their exp_comp_t as the simulation time expands

# 实例化MyGraph
def instantiate_mygraph_from_json(json_path, time_norm_factor):
    srcs, ops, sinks, task_attr, src_attr, sink_attr, pid2name = load_graph_from_json(json_path, time_norm_factor)
    G = MyGraph(srcs, ops, sinks, task_attr, src_attr, sink_attr)
    return G, pid2name

def instantiate_processors(partition_cfg: PartitionConfig, event_t, policy:str=None, stat_param:dict={}):
    # 创建传感器处理器，使用动态映射模式
    G = partition_cfg.G
    src_nodes = [n for n, x in G.logical_graph.in_degree() if x == 0]
    stats_collector = StatisticsCollector(**stat_param)
    # record the non-src task count
    stats_collector.set_task_cnt(
        len([
            n_ for n_ in G.logical_graph.nodes 
                if G.logical_graph.nodes[n_]['type'] == "op"
        ]))
    sen_p0 = Sen_p("sen_p", 3, 1, G, set(src_nodes), stats_collector)
    acc_p_list = acc_p_factory(policy, partition_cfg, stats_collector)
    processors = [sen_p0] + acc_p_list
    for i, proc in enumerate(processors):
        print(f"Processor {i}: {proc}")
    event_t = GlobalEvent_t(event_t)
    return processors, event_t, stats_collector

def initialize_events(policy: str, G: MyGraph, TSMap_list: list) -> set:
    """
    Initializes the event set based on the scheduling policy.
    """
    event_t = set()
    # 1. Common events: all policies have external events from sensor triggers
    for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "src"]:
        t = elim_nume_error(G.logical_graph.nodes[node]['offset'])
        event_t.add((t, "external"))
        print_if_verbose(f"SRC node {node}'s 0th event at {t}")

    # 2. Policy-specific events
    if policy == "cyc":
        # Cyclic scheduling has table-driven events
        for TSmap in TSMap_list:
            if TSmap:
                for t, cfg in TSmap:
                    event_t.add((t, "table"))
        print_if_verbose(f"Initialized table events for CYC policy.")

    elif policy == "reserv":
        # Reservation-based scheduling also considers earliest release times (ert)
        for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "op"]:
            t = elim_nume_error(G.logical_graph.nodes[node]['ert'])
            event_t.add((t, "external")) # Can be treated as an external trigger
            print_if_verbose(f"OP node {node}'s ert event at {t}")
    
    # 'pglb' does not have additional specific events initially.
    
    return event_t
