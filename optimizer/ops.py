import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import os
smoke_test = ('CI' in os.environ)


# clamp_min, minimum, maximum, ceil，>=, <=, ==

ceil_factor = 10
comp_scale = 10
eq_scale = 10

clamp_min_diff = lambda x, min_val: torch.nn.functional.softplus(x - min_val) + min_val
minimum_diff = lambda x, y: x - torch.nn.functional.softplus(x - y)
maximum_diff = lambda x, y: x + torch.nn.functional.softplus(y - x)
ceil_diff = lambda x: x + (1 - torch.sigmoid(ceil_factor * (x - torch.floor(x) - 0.5)))
ge_diff = lambda x, y: torch.sigmoid(comp_scale * (x - y))
le_diff = lambda x, y: 1 - torch.sigmoid(comp_scale * (y - x))
eq_diff = lambda x, y: torch.exp(-eq_scale * (x - y)^2)

import zuko
zuko.flows.PlanarFlow

OP_MAP_base = {
    'clamp_min': torch.clamp_min,
   'minimum': torch.minimum,
   'maximum': torch.maximum,
    'ceil': torch.ceil,
    'ge': torch.ge,
    'le': torch.le,
    'eq': torch.eq,
}

OP_MAP_diff = {
    'clamp_min': clamp_min_diff,
   'minimum': minimum_diff,
   'maximum': maximum_diff,
    'ceil': ceil_diff,
    'ge': ge_diff,
    'le': le_diff,
    'eq': eq_diff,
}


def update_w_rem(
                w_prev, q_prev, # step-dependent variable
                rate_c, # step-independent variable
                delta_T, # constant
                 ):
    """
    Get the executed load in previous time step
        math:: \hat w^i_t = \max(w^i_{t-1} - q^i_{t-1} rate^i_{t-1} \delta T, 0)

    Args:
        w_prev: remaining load at the beginning of the previous time step.
        q_prev: allocated resources at the beginning of the previous time step
        rate_c: rate of execution w.r.t. the allocated resources
        delta_T: timestep size

    Returns:
        w_rem: the remiaining load at the beginning of the current time step.
    """
    processed = q_prev* rate_c * delta_T 
    w_rem = torch.clamp_min(w_prev - processed, 0)
    return w_rem

def update_IF(IA_prev, w_rem, # step-dependent variable
              ft_c, # step-independent variable
              t_curr, is_source_node, ld_finish_threshold=1e-3 # constant
              ):
    """
    Update the finish indicator based on the executed load and the arrival load.
    math:: 
    \mathbb{IF}^i_t = \mathbb{IA}^i_{t-1} * \left\{\begin{IEEEeqnarraybox}[\relax][c]{l's}
        ft_i <= t & Sources\\
        \hat w^i_{t} == 0 & DNNs, 
    \end{IEEEeqnarraybox}\right.
    """
    IF_curr = IA_prev*torch.where(is_source_node, torch.ge(t_curr, ft_c), torch.ge(ld_finish_threshold, w_rem))
    return IF_curr

release_cond_fn = lambda t,r: (t>=r).float()

def update_IA(IF_curr, # step-dependent variable
              release_cond, # step-independent variable
              adj_mat, in_degree, epsilon=1e-12, # constant
              ):
    """
    math:: \mathbb{IA}^i_t = \min\limits_{i' \in pred.} \mathbb{IF}^{i'}_{t} 

    Args:
        IF_curr: indicator of whether the task is active at t 
        adj_mat: adjacency matrix of the task graph 
        in_degree: in-degree of each task node 
        release_cond: release condition of each task node 
        epsilon: small value to avoid numerical issues 

    Returns:
        _type_: _description_
    """
    # # 计算激活状态IA
    # pred_mask = torch.zeros_like(IA_prev)  # states of predecessors
    # for i in range(self.num_mids):
    #     # 构建前驱状态张量
    #     pred_status = torch.stack([
    #         IF_curr[self.node_map[p]] for p in self.graph.predecessors(i)
    #     ], dim=-1)  # (..., num_pred)
    #     pred_mask[i+self.num_src] = pred_status.min(dim=-1)  # (...,)            
    
    # 合并源节点和中间节点的激活条件
    # [83, 83] @ [4,4,4, 1, 83] -> [4,4,4, 1, 83]
    IA_curr = (torch.matmul(adj_mat, IF_curr.unsqueeze(-1)).squeeze(-1) - in_degree + epsilon) * release_cond
    return IA_curr

# exact bijective fun
def update_delta_w(
    IA_prev, IA_curr, # step-dependent variable
    ld_c # step-independent variable
    ):
    """
    Determine the arrival load before the start of the current time step.
    
    math:: \Delta w^i_t = ld^i_c * (\mathbb{IA}^i_t \oplus \mathbb{IA}^i_{t-1})
    
    Args:
        IA_prev: indicator of whether the task is active at t-1 
        IA_curr: indicator of whether the task is active at t 
        ld_c: the triggerd load given the active status at t
        delta_T: timestep size

    Returns:
        delta_w: the arrival load before the start of the current time step.
    """
    # rising edge detection
    return (1 - IA_prev) * IA_curr * ld_c


def update_alloc(
    w_curr, # step-dependent variable
    R_s, t, e_i, r_i # constant
    ):
    # min(max(r_i, q_min), q_max)
    slack = torch.clamp_min(e_i - t, 0) + 1e-6  # avoid zero division
    core_req = torch.ceil(w_curr / slack) # minimal resource required
    # minimal resource required
    q_min = torch.maximum(r_i, core_req)
    
    # calculate q_max, task by task
    # R_s - q_min_s[0:0].sum(), R_s - q_min_s[0:1].sum(), R_s - q_min_s[0:2].sum(), ...
    q_max_s = R_s - q_min.cumsum(dim=-1) + q_min
    q_max_s = torch.clamp_min(q_max_s, 0)
    q_min = torch.minimum(q_min, q_max_s)
    return q_min 

# define the baseline model
# import pyro.distributions.transforms as T

# define a bijective approximation 




def update_IF_mid(
              ft_c, # step-independent variable
              t_curr # constant
              ):
    IF_curr = (t_curr >= ft_c)
    return IF_curr

def update_IF_src(IA_prev, w_rem, # step-dependent variable
              ld_finish_threshold=1e-3 # constant
              ):
    IF_curr = IA_prev*(w_rem <= ld_finish_threshold)
    return IF_curr