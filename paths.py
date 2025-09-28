from dataclasses import dataclass, field
import os
from typing import Optional
from global_var import (
  bin_save_fmt, routing_table_save_fmt, cache_root_fmt, bin_fn_fmt,
  trace_fn_w_seed_fmt, trace_fn_wo_seed_fmt,
  plt_fn_w_seed_fmt, plt_fn_wo_seed_fmt,
  csv_root_fmt, trace_root_fmt, plot_root_fmt, cfg_root_fmt,
  log_root_fmt, cfg_root_dir_fmt, ld_fmt
)


@dataclass
class PathContext:
    # basic path settings
    # type dir: cfg_dir, log_dir, plot_dir, trace_dir, cache_dir, csv_dir

    root_dir: str # related to cases and policies
    case: str # test case: bin_pack_new, cyclic, dynamic, glb_dyn, pglb 
    num_bins: int # Bin settings
    
    # load parameters
    aux_scale_factor: int
    e2e_latency: float
    
    # file suffix
    file_suffix: str 
    i_file_suffix: str 
    force_suffix: str 

    # compensation parameters, affect core count
    exec_t_comp_ratioA: float
    lateness_mode: str 

    num_cores: int
    exec_t_comp_ratioB: float
        
    # runtime parameters
    seed: int
    jitter: bool
    
    # static paths    
    # automatically generated config name and path, not initialized 
    # type_dir, root_dir, cfg_n
    csv_root: str = field(init=False)
    cache_root: str = field(init=False)
    trace_root: str = field(init=False)
    plot_root: str = field(init=False)
    log_root: str = field(init=False)
    cfg_root_dir: str = field(init=False)
    graph_fn: str = field(init=False)
    cfg_n: str = field(init=False)
    

    def __post_init__(self):
        """generate static paths, reduce dependencies on utils.py"""
        self.refresh_config()

    def _ensure_dir_exists(self, path: str) -> None:
        """确保路径的目录存在，如果不存在则创建。"""
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def refresh_config(self):
        """根据影响 cfg_n 的字段刷新配置名与静态路径。"""
        cfg_para_dict = {
            "exec_t_comp_ratioA": self.exec_t_comp_ratioA,
            "lateness_mode": self.lateness_mode,
        }
        para_scan_group1 = {
            "aux_scale_factor": self.aux_scale_factor,
            "e2e_latency": self.e2e_latency
        }
        self.cfg_n = cfg_root_fmt.format(**cfg_para_dict, **para_scan_group1)
        self.graph_fn = ld_fmt.format(**para_scan_group1)

        # init static paths
        self.cache_root = cache_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.trace_root = trace_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.plot_root = plot_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.log_root = log_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.cfg_root_dir = cfg_root_dir_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.csv_root = csv_root_fmt.format(root_dir=self.root_dir)
        self.graph_fn = os.path.join(self.cache_root, f"graph_{self.graph_fn}.json")

        # create path folders
        for path in [self.cache_root, self.trace_root, self.plot_root, self.log_root, self.cfg_root_dir, self.csv_root]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    # ---------- Path helpers on context ----------
    def get_bin_list_path(self) -> str:
        params = {
            "root_dir": self.root_dir,
            "cfg_n": self.cfg_n,
            "force_suffix": self.force_suffix,
            "num_cores": self.num_cores,
            "i_file_suffix": self.i_file_suffix,
        }
        path = bin_save_fmt.format(**params)
        self._ensure_dir_exists(path)
        return path

    def get_routing_table_path(self) -> str:
        params = {
            "root_dir": self.root_dir,
            "cfg_n": self.cfg_n,
            "num_cores": self.num_cores,
            "i_file_suffix": self.i_file_suffix,
            "force_suffix": self.force_suffix,
        }
        path = routing_table_save_fmt.format(**params)
        self._ensure_dir_exists(path)
        return path

    def get_bin_fn_regex(self) -> str:
        return bin_fn_fmt.format(num_cores=r"(\d*)", force_suffix=self.force_suffix, i_file_suffix=self.i_file_suffix)

    def get_trace_path(self) -> str:
        fmt = trace_fn_w_seed_fmt if self.jitter else trace_fn_wo_seed_fmt
        params = {
            "trace_root": self.trace_root,
            "case": self.case,
            "force_suffix": self.force_suffix,
            "num_cores": self.num_cores,
            "seed": self.seed,
            "file_suffix": self.file_suffix,
        }
        path = fmt.format(**params)
        self._ensure_dir_exists(path)
        return path

    def get_plot_path(self, plt_size: str, case: Optional[str] = None, with_seed: Optional[bool] = None) -> str:
        fmt = plt_fn_w_seed_fmt if (self.jitter if with_seed is None else with_seed) else plt_fn_wo_seed_fmt
        params = {
            "plot_root": self.plot_root,
            "case": case or self.case,
            "plt_size": plt_size,
            "force_suffix": self.force_suffix,
            "num_cores": self.num_cores,
            "seed": self.seed,
            "file_suffix": self.file_suffix,
        }
        path = fmt.format(**params)
        self._ensure_dir_exists(path)
        return path

    def get_csv_path(self, name: str) -> str:
        path = os.path.join(self.csv_root, name)
        self._ensure_dir_exists(path)
        return path

    def get_log_path(self, case: Optional[str] = None) -> str:
        case_name = case or self.case
        filename = f"{case_name}_{self.force_suffix}{self.num_cores}{self.file_suffix}.log.txt"
        path = os.path.join(self.log_root, filename)
        self._ensure_dir_exists(path)
        return path
    
    def get_stat_log_path(self, case: Optional[str] = None) -> str:
        case_name = case or self.case
        filename = f"stat_{case_name}_{self.force_suffix}{self.num_cores}{self.file_suffix}.txt"
        path = os.path.join(self.log_root, filename)
        self._ensure_dir_exists(path)
        return path

    def get_cfg_path(self, cfg_name: str) -> str:
        path = os.path.join(self.cfg_root_dir, cfg_name)
        self._ensure_dir_exists(path)
        return path
