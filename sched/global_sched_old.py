"""global_sched_old.py — moved dead code from global_sched.py (B3-MOVE-007 (reclassified unused→old)).
Contents: naive_iso (isolation algo old version, 0 callers direct+indirect).
Moved, not rewritten. Archive only. Recover: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from typing import List, Dict
    from sched.scheduling_table import SchedulingTableInt

import math
from collections import OrderedDict, defaultdict
import numpy as np
from global_var import *
from model.resource_agent import Resource_model_int
from model.task_queue_agent import TaskQueue 
from model.buffer import Buffer
from model.message.msg_dispatcher import MsgDispatcher
from model.message.message_handler import message_trigger_event_new
from model.streaming_processing.wartermark_strategy import WatermarkStrategy
from model.message.data_pipe import DataPipe
from model.position_table import PosTableInt
from model.performance import slack_comp

from sched.monitor_agent import Monitor
from sched.scheduler_agent import Scheduler, check_miss, check_complete
from sched.scheduler_agent import data_pipe_read, pendingToReady
from sched.pre_alloc_new import glb_alloc_new2
from sched.bin_ops import new_bin, get_initlist_and_biniter, static_1_bin
from sched.sort_function import get_process_sort
from sched.packing_solver.gurobi_MP_semi2DClst import ClusterGurobiSolverSemi2D
from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id
from sched.bin_ops import bin_iter_list
from networkx import DiGraph
from functools import reduce
from sched.slack_estim import get_chains
from sched.scheduling_table import init_event
from sched.binpack_config import BinPackConfig

# 默认配置实例（使用 BinPackConfig 包装器）
# 注意：这些默认值主要作为函数签名的 fallback，实际运行时由 input_parser 从 JSON 文件加载
default_binpack_cfg = BinPackConfig()
# ==================== top-level scheduling procedure ====================

# === B3-MOVE-007 (reclassified unused→old): naive_iso (dead) ===
def naive_iso(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
        ):
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(hyper_p//timestep)
    # assert math.isclose(hyper_p, tab_temp_size*timestep, abs_tol=numerical_error_tol_abs), \
    #         "hyper_p should be the multiple of timestep"
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores
    # glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    from task.task_cfg import pre_assign_priority
    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
        bin_list, glb_p_list, 0, 
        _new_bin, "all_isolation", pre_assign_priority)
    print(bin_list)
