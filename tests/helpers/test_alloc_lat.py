from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from approach.approach_initiator import instantiate_processors, instantiate_mygraph_from_json
from approach.approach_sched import PartitionConfig
from approach.approach_sim import run_simulation
from approach.approach_Eq import set_time_unit, trasfer_realloc_as_task
from global_var import BW_DRAM, GLB_BUFFER_SIZE_PER_CORE, FLOPS_PER_CORE
from approach.approach_def import set_verbose_output

set_verbose_output(False)
time_unit, time_norm_factor = set_time_unit(1e-6, False)

# 加载图
graph_json_pth = REPO_ROOT / 'cache/coalescing_scan/n_bins_max/x1_0.1s_rda-99.00%(S)_ignore/graph_x1_0.1s.json'
G, pid2name = instantiate_mygraph_from_json(graph_json_pth, time_norm_factor=time_norm_factor)

# 配置
swt_lat = trasfer_realloc_as_task(BW_DRAM, 400, GLB_BUFFER_SIZE_PER_CORE, time_norm_factor)
sink_and_op_nodes = [n for n, x in G.logical_graph.in_degree() if x > 0]
partition_cfg = PartitionConfig(
    num_partitions=1,
    cap_list=[400],
    base_pwr_list=[FLOPS_PER_CORE],
    mapped_node_list=[sink_and_op_nodes],
    TSmap_list=[None],
    swt_lat_list=[swt_lat],
    G=G,
    T_hp=0.1,
)

# 创建事件
from approach.approach_Eq import elim_nume_error
event_t = set()
for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == 'src']:
    t = elim_nume_error(G.logical_graph.nodes[node]['offset'])
    event_t.add((t, 'external'))

# 创建处理器并运行
processors, event_t, stats_collector = instantiate_processors(partition_cfg, list(event_t), policy='pglb')
run_simulation(processors, event_t, G, num_hp=10, T_hp=0.1, verbose=False, var_en=False)

# 打印结果
info = stats_collector.get_sched_overhead_info()
print(info)