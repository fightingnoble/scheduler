import networkx as nx
from graphviz import Digraph

import torch
import pyro.poutine as poutine

from global_var import *
from model.event_gen.e2e_latency import get_truncnorm_para
from utils import core_distr
from optimizer.dist_custom import TruncatedNormal

class SchedulerBase:
    def __init__(self, graph:nx.DiGraph, 
                 M = 256,
                 T_hp = 0.1, num_hp = 1, delta_T = 1e-4,
                 alpha = 1.0, S_max = 20,
                #  ref_value = None, ft_scale = None, ld_scale = None,
                 n_warm_up = 1, n_drain = 1, 
                 max_Categorical_num = 10, 
                 ):
        
        src_cache = [n for n, degree in dict(graph.in_degree()).items() if degree == 0]
        # sort the src nodes by 1. offset, 2. their id
        src_cache.sort(key=lambda x: (graph.nodes[x]['offset'], x))
        sink_cache = [n for n, degree in dict(graph.out_degree()).items() if degree == 0]
        mid_cache = [n for n in graph.nodes if n not in src_cache and n not in sink_cache]
        self.graph = graph
        self.num_mids = len(mid_cache) # N_mid
        self.num_src = len(src_cache) # N_src
        self.num_sink = len(sink_cache) # N_sink
        self.num_tasks = self.num_src + self.num_mids  # Only source and middle nodes are tasks


        self.node_list = src_cache + mid_cache + sink_cache
        self.node_map = {n:i for i,n in enumerate(self.node_list)}
        adj_mat = torch.zeros(self.num_tasks, self.num_tasks)
        for node in src_cache + mid_cache:
            for pred in self.graph.predecessors(node):
                adj_mat[self.node_map[node], self.node_map[pred]] = 1
        self.adj_mat = adj_mat.detach().clone().requires_grad_(True)
        self.in_degree = torch.sum(adj_mat, dim=1)
        
        self.M = M          # 总资源数
        self.alpha = alpha  # overhead parameter
        self.S_max = S_max # maximum number of queues

        self.T_hp = T_hp  # hyper-period length
        self.delta_T = delta_T # time step size
        self.num_hp_sim = n_warm_up + n_drain + num_hp  # number of hyper-periods used for simulation
        self.num_hp_sti = n_warm_up + num_hp  # number of hyper-periods used for stimulates generation
        self.T = T_hp * num_hp # simulation time range
        self.hp_steps = int(T_hp / delta_T)  # number of time steps
        # ensure divisible by delta_T
        assert (self.hp_steps * self.delta_T - self.T_hp) < 1e-6
        self.sim_step = self.hp_steps * self.num_hp_sim  # total number of simulation steps
        self.cat_num = max_Categorical_num
        
        self.is_source_node = torch.cat([torch.full((self.num_src,), True), torch.full((self.num_mids, ), False)])
        # pre-define parameters for each node: 
        # trigger threshold, deadline, priority parameter, resource request thresholds, mapping
        
        # trigger threshold
        trigger_offset = [] 
        trigger_offset_flag = [] 
        timer = set()
        for i in range(self.num_src):
            trigger_offset.append(self.graph.nodes[self.node_list[i]]['offset'])
            timer.add(self.graph.nodes[self.node_list[i]]['offset'])
            trigger_offset_flag.append(True)
        timer.add(self.sim_step * delta_T)
        for i in range(self.num_mids):
            trigger_offset.append(-1)
            trigger_offset_flag.append(False)
        self.trigger_offset_flag = torch.tensor(trigger_offset_flag)
        self.trigger_offset = torch.tensor(trigger_offset)
        self.timer_t = torch.tensor(sorted(timer)) # default float32 
        
        # deadline threshold
        # all nodes are pre-defined with a default deadline 
        self.ddl_cache = torch.full((self.num_src + self.num_mids,), self.sim_step * delta_T)
        for i, node in enumerate(mid_cache):
            ddl = self.sim_step * delta_T
            for succ in self.graph.successors(node):
                if succ in sink_cache:
                    ddl = min(ddl, self.graph.nodes[succ]["ddl"] + self.graph.nodes[succ]['offset'])
                    # print(node, ddl, self.graph.nodes[succ]["ddl"]+ self.graph.nodes[succ]['offset'], self.graph.nodes[succ]["ddl"], self.graph.nodes[succ]['offset'])
            self.ddl_cache[i+self.num_src] = ddl
        
        # priority threshold
        self.priority_cache = torch.cat([torch.arange(self.num_src, dtype=torch.float), torch.ones(self.num_mids)*-1.])
        self.priority_cache_flag = torch.cat([torch.ones(self.num_src, dtype=torch.bool), torch.zeros(self.num_mids, dtype=torch.bool)])

        # resource request threshold 
        rsc_req_cache = []
        for i in range(self.num_src):
            rsc_req_cache.append(1.)
        for i in range(self.num_mids):
            rsc_req_cache.append(-1.)
        self.rsc_req_cache = torch.tensor(rsc_req_cache)
        
        # pre-defined mapping
        partition_sel = []
        partition_sel_flag = []
        unique_partition = {}
        for i in range(self.num_src):
            part_name = self.graph.nodes[self.node_list[i]]['partition']
            if part_name not in unique_partition:
                unique_partition[part_name] = len(unique_partition)
            partition_sel.append(unique_partition[part_name])
            partition_sel_flag.append(True)
        
        for i in range(self.num_mids):
            partition_sel.append(-1)
            partition_sel_flag.append(False)
        self.partition_sel_flag = torch.tensor(partition_sel_flag)
        self.partition_sel = torch.tensor(partition_sel)
        self.fixed_partition_num = len(unique_partition)
        
        # Define the parameters of the prior distributions
        # ft_std, ft_ref for source nodes
        # ld_values, ld_probs for middle nodes
        sen_period = []
        ft_ratio = []
        ft_offset = []
        
        for i in range(self.num_src):
            sen_period.append(1/self.graph.nodes[self.node_list[i]]['freq'])
            # ft_ratio.append(self.graph.nodes[self.node_list[i]]['comp_ratio'])
            # ft_offset.append(self.graph.nodes[self.node_list[i]]['comp_ratio']/self.graph.nodes[self.node_list[i]]['freq'])
        sen_period = torch.tensor(sen_period)
        ft_ratio = torch.tensor(ft_ratio)
        ft_ref = torch.tensor(ft_offset)
        _,ft_std,_,ft_a,ft_b = get_truncnorm_para(range_max=sen_period, jitter_sim_para={"scale": ft_ratio})
        
        # get flops, var_factor attributes for middle nodes
        self.ft_dist = TruncatedNormal(ft_ref, ft_std, torch.tensor(ft_a), torch.tensor(ft_b))
        self.ld_values, self.ld_probs = self.build_cat_prob_list()
                    
    def build_cat_prob_list(self):
        # Hard code to define a descrite load distribution 
        # truncated poisson distribution
        lambda_ld = 1
        ld_probs = [] 
        ld_values = []
        self.ld_mask = torch.cat([torch.full((self.num_src,), True), torch.full((self.num_mids, ), False)])

        for i in range(self.num_src):
            ld_probs.append(torch.tensor([1.0]))
            ld_values.append(torch.tensor([0.0]))

        ld_standard = torch.tensor([self.graph.nodes[self.node_list[i]]['flops']/FLOPS_PER_CORE for i in range(self.num_src, self.num_tasks)])
        for i in range(self.num_src, self.num_tasks):
            # middle nodes
            var_factor_list = self.graph.nodes[self.node_list[i]]['var_factor']
            # var_factor is now a list, get the length
            k = len(var_factor_list)
            # 0, 1, ..., k-1 
            assert k <= self.cat_num
            # poisson_cdf(i)/sum_i^{k+1} poisson_cdf(i) 
            import scipy.stats as stats
            # ***********************************************************************
            # temporary hard code logic to interpret var_factor
            assert k>0 
            # if k == 1, mean the node has no variance, the only legal value is 1x load
            # we mark the node in the ld_mask 
            if k==1: 
                ld_probs.append(torch.tensor([1.0]))
                ld_values.append(torch.tensor([ld_standard[i-self.num_src]]))
                self.ld_mask[i] = True
            else: 
                ini_probs = torch.zeros(k)
                for j in range(k):
                    ini_probs[j] = stats.poisson.pmf(j,lambda_ld)
                ld_probs.append(ini_probs/ini_probs.sum())
                ld_values.append(torch.tensor(var_factor_list, dtype=torch.float)*ld_standard[i-self.num_src])
        return ld_values, ld_probs

    def quant_t_e_r(self, x, scale):
        temp = x * scale
        with torch.no_grad():
            x_q = (x * scale).round()
            x_q = x_q.clamp(min=0, max=scale) - temp
        x_q += temp
        return x_q

    def redist_R_s(self, R_s_percentage, M):
        # retain gradients
        temp_R_s = R_s_percentage * self.M
        S_max = len(R_s_percentage)
        
        with torch.no_grad():
            # redistribute
            R_s_dict = {i:0 for i in range(S_max)}
            core_distr(R_s_dict, score_dict={i:R_s_percentage[i] for i in range(S_max)}, curr_aval_rsc=M, order_fn=lambda x:(x[1], -x[0]), sort=True)
            R_s_int = torch.tensor([R_s_dict[i] for i in range(S_max)])
            R_s = R_s_int - temp_R_s
        R_s += temp_R_s 
        return R_s


def build_cat_prob_tensor(self):
    lambda_ld = 1
    ld_probs = []    
    # the mask that indicates the nodes with no variance
    self.ld_mask = torch.cat([torch.full((self.num_src,), True), torch.full((self.num_mids, ), False)])
    for i in range(self.num_tasks):
        # source nodes
        if i < self.num_src:
            ld_probs.append(torch.ones(self.cat_num, dtype=torch.float)/self.cat_num)
            continue
        # middle nodes
        var_factor_list = self.graph.nodes[self.node_list[i]]['var_factor']
        # var_factor is now a list, get the length
        k = len(var_factor_list)
        # 0, 1, ..., k-1 
        assert k <= self.cat_num
        # poisson_cdf(i)/sum_i^{k+1} poisson_cdf(i) 
        ini_probs = torch.zeros(self.cat_num)
        import scipy.stats as stats
        # ***********************************************************************
        # temporary hard code logic to interpret var_factor
        assert k>0 
        # if k == 1, mean the node has no variance, the only legal value is 1x load
        # we mark the node in the ld_mask 
        if k==1: 
            ini_probs[0] = 1.
            self.ld_mask[i] = True
        else: 
            for j in range(k):
                ini_probs[j] = stats.poisson.pmf(j,lambda_ld)
        # ***********************************************************************
        ld_probs.append(ini_probs/ini_probs.sum())
    self.ld_probs = torch.stack(ld_probs)
            
    cat_values = torch.arange(self.cat_num, dtype=torch.float).expand(self.num_tasks, -1)
    return cat_values
    # self.plate_queues = pyro.plate("queues", self.S_max, dim=-2)


def draw_computational_graph(model):
    # 运行模型并获取跟踪
    trace = poutine.trace(model).get_trace()
    
    # 创建Graphviz对象
    dot = Digraph(comment='Scheduling Process', 
                 graph_attr={'rankdir': 'LR', 'nodesep': '0.5'})
    
    # 定义节点样式
    styles = {
        'sample': {'shape': 'ellipse', 'color': '#FFDDDD', 'style': 'filled'},
        'param': {'shape': 'box', 'color': '#DDDDFF', 'style': 'filled'},
        'deterministic': {'shape': 'diamond', 'color': '#DDFFDD', 'style': 'filled'}
    }
    
    # 添加所有节点
    nodes = set()
    for name, site in trace.nodes.items():
        if site["type"] == "param":
            dot.node(name, f"Param: {name}\n{site['value'].shape}", **styles['param'])
        elif site["type"] == "sample":
            dot.node(name, f"Sample: {name}\nDist: {site['fn']}", **styles['sample'])
        else:
            dot.node(name, f"Det: {name}", **styles['deterministic'])
        nodes.add(name)
    
    # 添加边关系
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            for child in site["cond_indep_stack"]:
                child_name = child.name
                if child_name in nodes:
                    dot.edge(child_name, name)
        if site["type"] in ["sample", "deterministic"]:
            for parent in site["args"]:
                if isinstance(parent, torch.Tensor) and parent.name in nodes:
                    dot.edge(parent.name, name)
    
    # 添加时间步展开示意
    with dot.subgraph(name='cluster_time_steps') as c:
        c.attr(label='Time Steps (Markov)', style='dashed')
        for t in range(3):
            c.node(f'step_{t}', f'Time Step {t}', shape='plaintext')
            if t > 0:
                c.edge(f'step_{t-1}', f'step_{t}', style='invis')
    
    return dot

