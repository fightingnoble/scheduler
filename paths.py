from dataclasses import dataclass, replace, field
import os
from typing import Optional
from global_var import (
  bin_save_fmt, routing_table_save_fmt, cache_root_fmt, bin_fn_fmt,
  trace_fn_w_seed_fmt, trace_fn_wo_seed_fmt,
  plt_fn_w_seed_fmt, plt_fn_wo_seed_fmt,
  csv_root_fmt, trace_root_fmt, plot_root_fmt, cfg_root_fmt,
  log_root_fmt, cfg_root_dir_fmt
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
    
    # compensation parameters, affect core count
    exec_t_comp_ratioA: float
    jitter_t_comp_ratio: float
    wsc_slack_ratio: float
    lateness_mode: str = "ignore"

    num_cores: Optional[int] = None
    exec_t_comp_ratioB: Optional[float] = None
        
    # runtime parameters
    seed: Optional[int] = None
    jitter: bool = False
    
    # file suffix
    file_suffix: str = ""
    i_file_suffix: str = ""
    force_suffix: str = ""
    
    # static paths
    
    # automatically generated config name and path, not initialized 
    # type_dir, root_dir, cfg_n
    csv_root: str = field(init=False)
    cache_root: str = field(init=False)
    trace_root: str = field(init=False)
    plot_root: str = field(init=False)
    log_root: str = field(init=False)
    cfg_root_dir: str = field(init=False)

    cfg_n: str = field(init=False)

    def __post_init__(self):
        """generate static paths, reduce dependencies on utils.py"""
        # 使用补偿比率参数生成配置名称
        cfg_para_dict = {
            "wsc_slack_ratio": self.wsc_slack_ratio,
            "exec_t_comp_ratioA": self.exec_t_comp_ratioA,
            "lateness_mode": self.lateness_mode,
            "jitter_t_comp_ratio": self.jitter_t_comp_ratio,
        }
        para_scan_group1 = {
            "aux_scale_factor": self.aux_scale_factor,
            "e2e_latency": self.e2e_latency
        }
        self.cfg_n = cfg_root_fmt.format(**cfg_para_dict, **para_scan_group1)
        
        # init static paths
        self.cache_root = cache_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.trace_root = trace_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.plot_root = plot_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.log_root = log_root_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.cfg_root_dir = cfg_root_dir_fmt.format(root_dir=self.root_dir, cfg_n=self.cfg_n)
        self.csv_root = csv_root_fmt.format(root_dir=self.root_dir)


class PathBuilder:
  def __init__(self, ctx: PathContext):
    self.ctx = ctx

  # 动态调整方法 - 处理扫描变量
  def with_num_cores(self, n: int):
    """设置线程池大小"""
    self.ctx = replace(self.ctx, num_cores=n)
    return self

  def with_num_bins(self, n: int):
    """设置bin数量"""
    self.ctx = replace(self.ctx, num_bins=n)
    return self

  def with_case(self, case: str):
    """设置测试用例"""
    self.ctx = replace(self.ctx, case=case)
    return self

  def with_jitter(self, flag: bool):
    """设置是否启用抖动"""
    self.ctx = replace(self.ctx, jitter=flag)
    return self

  def with_seed(self, seed: int):
    """设置随机种子"""
    self.ctx = replace(self.ctx, seed=seed)
    return self

  def with_compensation_ratios(self, exec_t_comp_ratioA: float = None, 
                              exec_t_comp_ratioB: float = None,
                              jitter_t_comp_ratio: float = None,
                              wsc_slack_ratio: float = None):
    """设置补偿比率参数"""
    updates = {}
    if exec_t_comp_ratioA is not None:
      updates['exec_t_comp_ratioA'] = exec_t_comp_ratioA
    if exec_t_comp_ratioB is not None:
      updates['exec_t_comp_ratioB'] = exec_t_comp_ratioB
    if jitter_t_comp_ratio is not None:
      updates['jitter_t_comp_ratio'] = jitter_t_comp_ratio
    if wsc_slack_ratio is not None:
      updates['wsc_slack_ratio'] = wsc_slack_ratio
    
    if updates:
      self.ctx = replace(self.ctx, **updates)
    return self

  def with_load_params(self, aux_scale_factor: int = None, e2e_latency: float = None):
    """设置负载参数"""
    updates = {}
    if aux_scale_factor is not None:
      updates['aux_scale_factor'] = aux_scale_factor
    if e2e_latency is not None:
      updates['e2e_latency'] = e2e_latency
    
    if updates:
      self.ctx = replace(self.ctx, **updates)
    return self

  def with_force_suffix(self, suffix: str):
    """设置强制后缀（用于force_num_cores）"""
    self.ctx = replace(self.ctx, force_suffix=suffix)
    return self

  def with_suffix(self, s: str):
    """添加文件后缀"""
    self.ctx = replace(self.ctx,
      file_suffix=f"{self.ctx.file_suffix}{s}",
      i_file_suffix=f"{self.ctx.i_file_suffix}{s}",
    )
    return self

  def for_repack(self, ratio: float):
    """为重新打包设置后缀"""
    return self.with_suffix(f"_ov_{ratio:.2f}_repack")

  def for_e2e(self, aux: int, e2e: float):
    """为端到端测试设置负载参数"""
    return self.with_load_params(aux, e2e)

  def for_scan_type(self, scan_type: str, **kwargs):
    """根据扫描类型设置参数"""
    if scan_type == "tp":
      # 线程池扫描
      if 'num_cores' in kwargs:
        self.with_num_cores(kwargs['num_cores'])
      if 'force_num_cores' in kwargs and kwargs['force_num_cores']:
        self.with_force_suffix("force_")
    elif scan_type == "bin":
      # Bin数量扫描
      if 'num_bins' in kwargs:
        self.with_num_bins(kwargs['num_bins'])
    elif scan_type == "ratioB":
      # 补偿比率B扫描
      if 'exec_t_comp_ratioB' in kwargs:
        self.with_compensation_ratios(exec_t_comp_ratioB=kwargs['exec_t_comp_ratioB'])
    return self

  # 路径生成方法 - 使用预生成的路径
  def bin_list_path(self) -> str:
    """生成bin列表文件路径"""
    params = {
      "root_dir": self.ctx.root_dir,
      "cfg_n": self.ctx.cfg_n,
      "force_suffix": self.ctx.force_suffix,
      "num_cores": self.ctx.num_cores,
      "i_file_suffix": self.ctx.i_file_suffix,
    }
    return bin_save_fmt.format(**params)

  def routing_table_path(self) -> str:
    """生成路由表文件路径"""
    params = {
      "root_dir": self.ctx.root_dir,
      "cfg_n": self.ctx.cfg_n,
      "num_cores": self.ctx.num_cores,
      "i_file_suffix": self.ctx.i_file_suffix,
      "force_suffix": self.ctx.force_suffix,
    }
    return routing_table_save_fmt.format(**params)

  def cache_root(self) -> str:
    """获取缓存根目录"""
    return self.ctx.cache_root

  def trace_root(self) -> str:
    """获取trace根目录"""
    return self.ctx.trace_root

  def plot_root(self) -> str:
    """获取plot根目录"""
    return self.ctx.plot_root

  def log_root(self) -> str:
    """获取log根目录"""
    return self.ctx.log_root

  def cfg_root_dir(self) -> str:
    """获取配置根目录"""
    return self.ctx.cfg_root_dir

  def csv_root(self) -> str:
    """获取CSV根目录"""
    return self.ctx.csv_root

  def bin_fn_regex(self) -> str:
    """生成bin文件名正则表达式"""
    return bin_fn_fmt.format(num_cores=r"(\d*)", force_suffix=self.ctx.force_suffix, i_file_suffix=self.ctx.i_file_suffix)

  def trace_path(self) -> str:
    """生成trace文件路径"""
    fmt = trace_fn_w_seed_fmt if self.ctx.jitter else trace_fn_wo_seed_fmt
    params = {
      "trace_root": self.ctx.trace_root,
      "case": self.ctx.case,
      "force_suffix": self.ctx.force_suffix,
      "num_cores": self.ctx.num_cores,
      "seed": self.ctx.seed,
      "file_suffix": self.ctx.file_suffix,
    }
    return fmt.format(**params)

  def plot_path(self, plt_size: str, case: Optional[str] = None, with_seed: Optional[bool] = None) -> str:
    """生成plot文件路径"""
    fmt = plt_fn_w_seed_fmt if (self.ctx.jitter if with_seed is None else with_seed) else plt_fn_wo_seed_fmt
    params = {
      "plot_root": self.ctx.plot_root,
      "case": case or self.ctx.case,
      "plt_size": plt_size,
      "force_suffix": self.ctx.force_suffix,
      "num_cores": self.ctx.num_cores,
      "seed": self.ctx.seed,
      "file_suffix": self.ctx.file_suffix,
    }
    return fmt.format(**params)


  def csv_path(self, name: str) -> str:
    """生成CSV文件路径"""
    return os.path.join(self.csv_root(), name)

  def log_path(self, case: Optional[str] = None) -> str:
    """生成log文件路径"""
    case_name = case or self.ctx.case
    filename = f"{case_name}_{self.ctx.force_suffix}{self.ctx.num_cores}{self.ctx.file_suffix}.log.txt"
    return os.path.join(self.ctx.log_root, filename)

  def cfg_path(self, cfg_name: str) -> str:
    """生成配置文件路径"""
    return os.path.join(self.ctx.cfg_root_dir, cfg_name)
