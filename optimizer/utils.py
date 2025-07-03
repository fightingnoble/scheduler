import networkx as nx
from graphviz import Digraph

import torch
import pyro.poutine as poutine

from global_var import *
from model.event_gen.e2e_latency import get_truncnorm_para
from utils import core_distr

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
        
        
        # get the sensor node's 1/freq *comp_ratio
        sen_period = []
        ft_ratio = []
        ft_offset = []
        
        for i in range(self.num_src):
            sen_period.append(1/self.graph.nodes[self.node_list[i]]['freq'])
            ft_ratio.append(self.graph.nodes[self.node_list[i]]['comp_ratio'])
            ft_offset.append(self.graph.nodes[self.node_list[i]]['offset'] + self.graph.nodes[self.node_list[i]]['comp_ratio']/self.graph.nodes[self.node_list[i]]['freq'])
        sen_period = torch.tensor(sen_period)
        ft_ratio = torch.tensor(ft_ratio)
        ft_ref = torch.tensor(ft_offset)
        _,ft_std,_,ft_a,ft_b = get_truncnorm_para(range_max=sen_period, jitter_sim_para={"scale": ft_ratio})
        
        # get flops, var_factor attributes for middle nodes
        self.ft_std = torch.cat([ft_std, torch.ones(self.num_mids)])
        self.ft_ref = torch.cat([ft_ref, torch.zeros(self.num_mids)])
        
        self.ld_values, self.ld_probs = self.build_cat_prob_list()
        
                
        # 遍历每个超周期和时间步
        self.is_source_node = torch.cat([torch.full((self.num_src,), True), torch.full((self.num_mids, ), False)])
        self.ddl_cache = torch.full((self.num_src + self.num_mids,), self.sim_step * delta_T)
        for i, node in enumerate(mid_cache):
            ddl = self.sim_step * delta_T
            for succ in self.graph.successors(node):
                if succ in sink_cache:
                    ddl = min(ddl, self.graph.nodes[succ]["ddl"] + self.graph.nodes[succ]['offset'])
                    # print(node, ddl, self.graph.nodes[succ]["ddl"]+ self.graph.nodes[succ]['offset'], self.graph.nodes[succ]["ddl"], self.graph.nodes[succ]['offset'])
            self.ddl_cache[i+self.num_src] = ddl

        # self.has_ddl = torch.tensor([0]*self.num_src + [graph.successors(n)[0] for n in mid_cache])
                    
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

        ld_standard = torch.tensor([self.graph.nodes[self.node_list[i]]['flops']/FLOPS_PER_CORE/self.delta_T for i in range(self.num_src, self.num_tasks)])
        for i in range(self.num_src, self.num_tasks):
            # middle nodes
            k = self.graph.nodes[self.node_list[i]]['var_factor']
            # 0, 1, ..., k 
            assert k+1 <= self.cat_num
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
                ini_probs = torch.zeros(k+1)
                for j in range(k+1):
                    ini_probs[j] = stats.poisson.pmf(j,lambda_ld)
                ld_probs.append(ini_probs/ini_probs.sum())
                ld_values.append(torch.arange(k+1, dtype=torch.float)*ld_standard[i-self.num_src])
        return ld_values, ld_probs

    def quant_t_e_r(self, x, scale):
        temp = x * scale
        with torch.no_grad():
            x_q = (x * scale).round()
            x_q = x_q.clamp(min=0, max=scale) - temp
        x_q += temp
        # extend t_i, r_i, e_i to include source nodes
        return torch.cat([torch.zeros(self.num_src), x_q])

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
        k = self.graph.nodes[self.node_list[i]]['var_factor']
        # 0, 1, ..., k 
        assert k+1 <= self.cat_num
        # poisson_cdf(i)/sum_i^{k+1} poisson_cdf(i) 
        ini_probs = torch.zeros(self.cat_num)
        import scipy.stats as stats
        # ***********************************************************************
        # temporary hard code logic to interpret var_factor
        assert k>0 
        # if k == 1, mean the node has no variance, the only legal value is 1x load
        # we mark the node in the ld_mask 
        if k==1: 
            ini_probs[1] = 1.
            self.ld_mask[i] = True
        else: 
            for j in range(k+1):
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

