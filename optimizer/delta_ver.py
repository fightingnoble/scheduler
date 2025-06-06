import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import Trace_ELBO
from graphviz import Digraph
from collections import OrderedDict
from global_var import *

from collections import defaultdict

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.distributions import constraints
import networkx as nx
from model.event_gen.e2e_latency import get_truncexpon_param, get_truncnorm_para

from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import torchsort
from pyro.ops.indexing import Vindex
from optimizer.utils import SchedulerBase
from pyro.distributions.util import broadcast_shape

ld_finish_threshold = 1e-3
state_avail_th = 0.5
epsilon = 1e-12
from optimizer.ops import (
    update_w_rem, update_IF, update_IA, update_delta_w, 
    update_alloc, release_cond_fn, update_IF_src, update_IF_mid
)

class SchedulingBayesNet(SchedulerBase):
    def __init__(self, graph:nx.DiGraph, 
                 M = 256,
                 T_hp = 0.1, num_hp = 1, delta_T = 1e-4,
                 alpha = 1.0, S_max = 20,
                #  ref_value = None, ft_scale = None, ld_scale = None,
                 n_warm_up = 1, n_drain = 1, 
                 max_Categorical_num = 10, 
                 max_plate_nesting = 1
                 ):
        
        
        super().__init__(graph, M, T_hp, num_hp, delta_T, alpha, S_max, n_warm_up, n_drain, max_Categorical_num)
        self.max_plate_nesting = max_plate_nesting                    
        # 初始化可学习参数
        # self.plate_tasks = pyro.plate("tasks", self.num_tasks, dim=-1)
        # self.plate_queues = pyro.plate("queues", self.S_max, dim=-2)
        
    def _init_parameters(self):
        """初始化所有可优化参数"""
        # 任务参数 (t_i, e_i, r_i, R_s_percentage, s_probs_mids)
        # 时间偏移参数（显式声明张量类型）
        
        # 0<=t_i<=1, 
        # need to be multiplied by T_hp to get the actual time
        # need greater than offset of source nodes, and less than ddl of sink nodes
        self.t_i_mid = pyro.param(
            "t_i",
            torch.rand(self.num_mids),
            constraint=constraints.unit_interval
        )
        # 0<=e_i<=1
        # need to be multiplied by T_hp to get the actual time
        # need greater than t_i, and less than ddl of sink nodes
        self.e_i_mid = pyro.param(
            "e_i",                                  # 优先级参数
            torch.rand(self.num_mids),
            constraint=constraints.unit_interval
        )

        # 0<=r_i<=M
        self.r_i_mid = pyro.param(
            "r_i",                                  # 预期资源需求
            torch.rand(self.num_mids)/2,
            constraint=constraints.unit_interval
        )
        # \sum_s R_s = M \forall s \in \{1, \ldots, S\}
        # 队列资源分配 (使用simplex约束)
        self.R_s_percentage = pyro.param(
            "R_s", 
            torch.ones(self.S_max)/self.S_max,
            constraint=constraints.simplex, 
        ) 

        # 队列分配 (Gumbel-Softmax松弛)
        # \sum_j s_ij = 1 \forall i \in \{1, \ldots, n\}
        self.s_probs_mids = pyro.param(
            "s_probs", 
            torch.ones(self.num_tasks, self.S_max)/self.S_max,
            constraint=constraints.simplex
        )
        
    @config_enumerate(default="parallel")
    def model(self): 
        self._init_parameters()
        ft_var = pyro.sample(f"ft", dist.Normal(0, self.ft_std).mask(self.is_source_node).to_event(1))
        ft = (self.ft_ref + ft_var) * self.is_source_node.float() 
        
        ld = [torch.tensor(0.0) for _ in range(self.num_src)]
        for i in range(self.num_src, self.num_tasks):
            if self.ld_mask[i]:
                ld_var = self.ld_values[i][0]
            else: 
                # print(i)
                ld_cat = pyro.sample(f"ld_{i}", dist.Categorical(probs=self.ld_probs[i]))
                # print(f"ld_cat: {ld_cat.shape}")
                ld_var = Vindex(self.ld_values[i])[ld_cat]
                # print(f"ld_var: {ld_var.shape}")
            ld.append(ld_var)
        # print("=====")

        # get the initial parameters of the system
        # (t_i, e_i, r_i, R_s_percentage, s_probs_mids)
        s_i_onehot_mid = pyro.sample(
            "s_i", 
            dist.RelaxedOneHotCategoricalStraightThrough(torch.tensor(0.5), probs=self.s_probs_mids).mask(~self.is_source_node).to_event(1)
            )  # temperature=0.5
        
        s_i_mid_t = s_i_onehot_mid[self.num_src:] @ torch.arange(self.S_max, dtype=torch.float) 
                                
        # sort the parameters by the priority (here is e_i, smaller is higher priority)
        # example: e_i = torch.tensor([3.0, 1.0, 2.0], requires_grad=True)
        # differentiable soft rank
        soft_ranks = torchsort.soft_rank(self.e_i_mid.unsqueeze(0), regularization_strength=1.0).squeeze(0)
        # print("Soft Ranks:", soft_ranks)  # get [2.0, 0.5, 1.5]

        # differential group sort
        sorted_indices_mid = torch.argsort(soft_ranks)
        # print("Sorted Indices:", sorted_indices)  # 输出 [1, 2, 0]
        t_i_mid = torch.gather(self.t_i_mid, -1, sorted_indices_mid)
        e_i_mid = torch.gather(self.e_i_mid, -1, sorted_indices_mid)
        r_i_mid = torch.gather(self.r_i_mid, -1, sorted_indices_mid)
        s_i_mid = torch.gather(s_i_mid_t, -1, sorted_indices_mid)

        # extend parameters to include source nodes
        t_i_q = self.quant_t_e_r(t_i_mid, self.hp_steps)
        e_i_q = self.quant_t_e_r(e_i_mid, self.hp_steps)
        r_i_q = self.quant_t_e_r(r_i_mid, self.M)
        R_s_q = self.redist_R_s(self.R_s_percentage, self.M)
        s_i_q = torch.cat([torch.full((self.num_src,), float(self.S_max)), s_i_mid]) 
        
        # The first trick is to broadcast. This works with or without enumeration.
        # get shape of the state space 
        
        # enumeration|batch|event 
        enum_bat_shape = broadcast_shape(*[i.shape for i in ld])
        state_shape= enum_bat_shape + (self.num_tasks,)
        
        # initial state
        IA_prev = torch.zeros(state_shape)  # 初始未激活
        w_prev =  torch.zeros(state_shape)   # 初始剩余负载为0
        q_prev =  torch.zeros(state_shape)   # 初始资源分配为0
                
        total_loss = torch.tensor(0.0)

        for t in pyro.markov(range(self.hp_steps, self.hp_steps + 2)):
            t_curr = t * self.delta_T  # Relative time within hyper-period

            # 计算激活状态IA
            w_rem_t = update_w_rem(w_prev, q_prev, 1, self.delta_T)
            IF_curr = update_IF(IA_prev, w_rem_t, ft, t_curr, self.is_source_node, ld_finish_threshold)
            IA_t = update_IA(IF_curr, release_cond_fn(t_curr, t_i_q), self.adj_mat, self.in_degree, epsilon=1e-12)
            
            delta_w_t = []
            for i in range(self.num_tasks):
                # first broadcast to state_shape
                # IA_prev_i = broadcast_shape(torch.select(IA_prev, -1, i), state_shape[])
                # event dim: tasks (i)
                # (enumeration|batch|event) -> (enumeration|batch)
                delta_w_i  = update_delta_w(torch.select(IA_prev, -1, i), 
                                                torch.select(IA_t, -1, i), ld[i])
                delta_w_t.append(delta_w_i.expand(enum_bat_shape))
            #  (enumeration|batch) -> (enumeration|batch|event)
            delta_w_t = torch.stack(delta_w_t, dim=-1)
            w_curr_t = w_rem_t + delta_w_t

            q_curr_t = torch.zeros(state_shape)
            for s in range(self.S_max): 
                mask = (s_i_q == s)
                q_curr_t += update_alloc(
                    w_curr_t * mask.float(),
                    R_s_q[s],
                    t_curr,
                    e_i_q,
                    r_i_q
                )
                
            
            # IA_curr = pyro.sample(f"IA_curr_{t}", dist.Delta(IA_t).to_event(1))
            # w_curr = pyro.sample(f"w_curr_{t}", dist.Delta(w_curr_t).to_event(1))
            # q_curr = pyro.sample(f"q_curr_{t}", dist.Delta(q_curr_t).to_event(1))


            
            # 记录目标函数项 -----------------------------------------------
            # 目标1: 超时惩罚
            for i in range(self.num_tasks):
                if t_curr >= self.ddl_cache[i]:
                    pyro.factor(f"obj1_term_{t}_{i}", -torch.select(w_curr_t, -1, i)) 
            
            # 目标2: 重调度惩罚
            # reschedule_cost = (q_curr != q_prev).float() * (q_curr + q_prev)**2
            # pyro.factor(f"obj2_term_{t}", -self.alpha * reschedule_cost.sum())
            # # 传递状态到下一步 ---------------------------------------------
            # IA_prev = IA_curr
            # w_prev = w_curr
            # q_prev = q_curr 
            IA_prev = IA_t
            w_prev = w_curr_t
            q_prev = q_curr_t

        IA_curr = pyro.sample(f"IA_curr_{t}", dist.Normal(IA_t, 1e-4).to_event(1))
        w_curr = pyro.sample(f"w_curr_{t}", dist.Normal(w_curr_t, 1e-4).to_event(1))
        q_curr = pyro.sample(f"q_curr_{t}", dist.Normal(q_curr_t, 1e-4).to_event(1))            
    def guide(self):
        pass

    def train(self, num_epochs=1000, lr=0.01):
        """训练过程"""
        optimizer = ClippedAdam({"lr": lr, "clip_norm": 10.0})
        
        svi = SVI(self.model, 
                #   self.guide,
                    config_enumerate(
                        AutoNormal(poutine.block(self.model, hide=[f"ld_{i}" for i, flg in enumerate(self.ld_mask) if not flg])), 
                        "parallel"
                        ),
                  optimizer, 
                  loss=TraceEnum_ELBO(max_plate_nesting=self.max_plate_nesting))

        losses = []
        for epoch in range(num_epochs):
            loss = svi.step()
            losses.append(loss)
            
            # 应用资源约束投影
            with torch.no_grad():
                # 保持R_s在simplex空间
                self.R_s_percentage.data = self._project_to_simplex(self.R_s_percentage.data)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return losses
    
    @staticmethod
    def _project_to_simplex(x):
        """投影到simplex约束"""
        sorted_x, _ = torch.sort(x, descending=True)
        cumsum = torch.cumsum(sorted_x, dim=0)
        k = torch.arange(1, x.size(0)+1).to(x.device)
        condition = sorted_x - (cumsum - 1.0) / k > 0
        rho = torch.max(k[condition])
        theta = (cumsum[rho-1] - 1.0) / rho
        return torch.relu(x - theta)
    
    def get_parameters(self):
        """获取解析后的参数"""
        t_i = self.quant_t_e_r(self.t_i_mid, self.hp_steps)
        e_i = self.quant_t_e_r(self.r_i_mid, self.M)
        r_i = self.quant_t_e_r(self.r_i_mid, self.M)
        s_i = torch.argmax(self.s_probs_mids.data, dim=1)
        R_s = self.redist_R_s(self.R_s_percentage, self.M)
        return {
            't_i': t_i,
            'r_i': e_i,
            'e_i': r_i,
            's_i': s_i,
            'R_s': R_s,
        }

    
# 使用示例
if __name__ == "__main__":
    # scheduler = SchedulingOptimizer()
    # losses = scheduler.train(num_epochs=1000)
    # params = scheduler.get_parameters()
    # print("Optimized Parameters:", params)

    from task.task_cfg import load_json_graph_utils
    from optimizer.utils import draw_computational_graph
    G = load_json_graph_utils('./cache/graph.json')
    max_plate_nesting = 1
    first_available_dim = -1 - max_plate_nesting

    inputs = {
        "graph": G, 
        "M": 256,
        "T_hp": 0.1, "num_hp": 1, "delta_T": 1e-4,
        "alpha": 1.0, "S_max": 20,
        # task def
        # "ref_value": 1/sensor_freq, "ft_scale": 0.3, "ld_scale": 1,
        # "n_warm_up": 1, "n_drain": 1, 
        "max_plate_nesting": max_plate_nesting,
    }
    model = SchedulingBayesNet(**inputs)
    dot = pyro.render_model(
        model.model, model_args=(),     
        render_params=True,
        render_distributions=True,
        render_deterministic=True
    )
    dot.render('plot/scheduling_graph', view=True)

    trace = poutine.trace(poutine.enum(model.model, first_available_dim=first_available_dim)).get_trace()
    trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
    print(trace.format_shapes())

    pyro.clear_param_store()
    model.train()
    # 生成并保存图形
    dot = draw_computational_graph(model.model)
    dot.format = 'png'
    dot.render('scheduling_graph', view=True)
    
    # 为什么在guide里面不枚举？
    # 为什么config_enumerate不起作用，什么时候应该用这个东西显性配置枚举？
    