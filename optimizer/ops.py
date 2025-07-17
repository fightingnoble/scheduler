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
le_diff = lambda x, y: torch.sigmoid(comp_scale * (y - x))
eq_diff = lambda x, y: torch.exp(-eq_scale * (x - y)^2)


OP_MAP_BASE = {
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
OP_MAP = OP_MAP_BASE

def update_w_rem(
                w_prev, q_prev, # step-dependent variable
                rate_c, # step-independent variable
                delta_T, # constant
                op_map=OP_MAP
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
    w_rem = op_map["clamp_min"](w_prev - processed, 0)
    return w_rem

def update_IF(IA_prev, w_rem, # step-dependent variable
              term_cond, 
              ld_finish_threshold=torch.tensor(1e-3), # constant
            op_map=OP_MAP
              ):
    """
    Update the finish indicator based on the executed load and the arrival load.
    math:: 
    \vec{\Isfinish}_t = \vec{\Isactive}_{t-1} (\vec{w}_t == 0)
    """
    IF_curr = op_map["maximum"](IA_prev*op_map["ge"](ld_finish_threshold, w_rem), term_cond)
    return IF_curr

release_cond_fn = lambda t,r,op_map=OP_MAP: op_map["ge"](t,r).float()
term_cond_fn = lambda t,ddl,op_map=OP_MAP: op_map["ge"](t,ddl).float()

def update_IA(IF_curr, # step-dependent variable
              release_cond, # step-independent variable
              is_source_node, 
              adj_mat, in_degree, epsilon=torch.tensor(1e-12), # constant
              op_map=OP_MAP
              ):
    """
    math::
        &\vec{\Isactive}_t = \left\{\begin{IEEEeqnarraybox}[\relax][c]{l's}
        \vec{at}_c == t & Sources\\
        \min\limits_{i' \in pred.} \Isfinish^{i'}_{t} & DNNs, 
        \end{IEEEeqnarraybox}\right.\\ 


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
    # condition, the satisfied predecessor equals to the in-degree of the node, 
    # is equivalent to that adj_mat@ IF_curr - in_degree> -epsilon
    IA_curr = torch.where(is_source_node, 
                           torch.ones_like(IF_curr), 
                          op_map['ge'](torch.matmul(adj_mat, IF_curr.unsqueeze(-1)).squeeze(-1), in_degree - epsilon)) * release_cond
    return IA_curr

# exact bijective fun
def update_delta_w(
    IA_prev, IA_curr, # step-dependent variable
    ld_c, # step-independent variable
    op_map=OP_MAP
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


def prior_alloc(
    R_s, r_i, # constant
    core_req, 
    op_map=OP_MAP
    ):
    # minimal resource required
    q_min = op_map["maximum"](r_i, core_req)
    # calculate q_max, task by task
    # R_s - q_min_s[0:0].sum(), R_s - q_min_s[0:1].sum(), R_s - q_min_s[0:2].sum(), ...
    q_max_s = R_s - q_min.cumsum(dim=-1) + q_min
    q_max_s = op_map["clamp_min"](q_max_s, 0)
    q_min = op_map["minimum"](q_min, q_max_s)
    return q_min 


def core_req_lb(r_i_q, w_curr, 
                ld_finish_threshold=torch.tensor(1e-3), # constant
                op_map=OP_MAP):
    return r_i_q * op_map["ge"](w_curr, ld_finish_threshold)

alloc_seq = lambda w_curr, t, e_i, op_map=OP_MAP: w_curr.new_zeros(w_curr.shape)

def alloc_spatial(w_curr, t, e_i, op_map=OP_MAP):
    slack = op_map["clamp_min"](e_i - t, 0) + 1e-6  # avoid zero division
    core_req = op_map["ceil"](w_curr / slack) # minimal resource required
    return core_req

def Next_comp_time(
    w_curr, # step-dependent variable
    q_curr, # step-dependent variable
    rate_c, # step-independent variable
    epsilon=torch.tensor(1e-3), # constant
    op_map=OP_MAP
):
    """
    Calculate the next computation time based on the remaining load and the allocated resources.
    math:: t_{comp,i,t+1} = \frac{\hat w^i_t}{q^i_t rate^i_t}

    Args:
        w_curr: remaining load at the beginning of the current time step.
        q_curr: allocated resources at the beginning of the current time step
        rate_c: rate of execution w.r.t. the allocated resources

    Returns:
        t_comp_next: the next computation time.
    """
    q_alloc = q_curr + epsilon  # avoid zero division
    t_comp_next = (w_curr / (q_alloc * rate_c))
    return t_comp_next

# define the baseline model
# import pyro.distributions.transforms as T

# define a bijective approximation 

