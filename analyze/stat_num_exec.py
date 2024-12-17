import os
import argparse
import pandas as pd
import re
from task.task_cfg import load_taskattrib, creat_logical_graph
from task.task_cfg import task_graph_srcs, task_graph_ops, task_graph_sinks
from sched.slack_estim import deduce_num_exec
from analyze.pattern import get_path_var_scaner
from analyze.pattern import folder_pattern, folder_pattern_keys, folder_type
from analyze.pattern import get_group_dict, get_log_regexp

# log content pattern
# (lateness detected)TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}!!
# \t\tTASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}!!

completed_pattern1 = r'\t\tTASK (\d+):([\w_]+)\((\d+)\) COMPLETED @ ([\d.]+)/([\d.]+)!!'
completed_pattern2 = r'\(lateness detected\)TASK (\d+):([\w_]+)\((\d+)\) COMPLETED @ ([\d.]+)/([\d.]+)!!'

force_completed_pattern = r'bin_split'

# TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!
miss_pattern = r'\t\tTASK (\d+):([\w_]+)\((\d+)\) MISSED DEADLINE @ ([\d.]+)/([\d.]+)!!'

# f"		{_p.task.name} triggered @ {ingestion_time:.6f}/{event_time:.6f}"
trigger_pattern = r'\t\t([\w_]+) triggered @ ([\d.]+)/([\d.]+)'
pattern = r'(?P<task>\w+)\s+triggered\s+@\s+(?P<time>\d+\.\d+)'

# file name pattern
# jitter_en_log_fn_pattern = r"(glb_dyn|dyn)(_\d+)?_jitter_en_seed_("+seed_re+r").log.txt"
false_jitter_en_log_fn_pattern = r"(glb_dyn|dyn)(_\d+)?_jitter_en.log.txt"


def extract_num_exec(profiling_filename, aux_scale_factor, n_p, warmup_dis, mode=""):
    """
    mode: e2e, or empty
    e2e: only count the number of execution of the sink nodes
    empty: count the number of execution of all nodes
    """
    taskattr_dict, f_gcd = load_taskattrib(profiling_filename, verbose=False) 
    num_exec = 0
    if aux_scale_factor != 1:
        for node, taskattr in taskattr_dict.items():
            # scale up the thread scaling factor
            if taskattr.timing_flag == "realtime":
                taskattr.thread_scaling_factor *= aux_scale_factor
    if mode == "e2e":
        logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
        # get the predecessor of sink nodes
        sink_predecessor = set()
        for node in task_graph_sinks:
            sink_predecessor.update(logical_graph_nx.predecessors(node))
        for node in sink_predecessor:
            taskattr = taskattr_dict[node]
            num_exec += deduce_num_exec(taskattr.freq, f_gcd, taskattr.thread_scaling_factor)
        num_exec = int(num_exec) * (n_p + (not warmup_dis))
        return num_exec, sink_predecessor
    else:
        for node in taskattr_dict.keys():
            taskattr = taskattr_dict[node]
            num_exec += deduce_num_exec(taskattr.freq, f_gcd, taskattr.thread_scaling_factor)
        num_exec = int(num_exec) * (n_p + (not warmup_dis))
        return num_exec

# extract number of cores
def extract_num_cores(filename):
    pattern = r'(?P<num_cores>\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group('num_cores'))
    else:
        return -1

def count_miss_comp(folder, output, 
                    log_pattern, log_pattern_keys, log_pattern_type,
                    get_ref_num_exec=False,
                    profiling_filename="", aux_scale_factor=0, n_p=0, warmup_dis=False,
                    verbose=False
                    ):

    completed_list = []  # 用于存储已完成任务及其计数
    miss_list = []  # 用于存储未完成任务及其计数
    filename_list = []  # 用于存储文件名
    trigger_list = []

    # 遍历指定文件夹下的所有 .log.txt 文件, 按照文件名排序
    # for filename in os.listdir(folder):
    force_list = []
    for filename in sorted(os.listdir(folder)):
        if re.match(false_jitter_en_log_fn_pattern, filename):
            continue
        if filename.endswith(".log.txt"):
            completed_dict = {}  # 用于存储已完成任务及其时间
            miss_dict = {}  # 用于存储未完成任务及其时间
            trigger_dict = {}
            file_path = os.path.join(folder, filename)
            with open(file_path, "r") as file:
                for line in file:
                    # if matched force_completed_pattern in line:
                    if "'algorithm': 'bin_split'" in line and "test_case='bin_pack_new'" in line:
                        print("Force completed: ", file_path)
                        force_list.append(filename)
                        break 
                    if "COMPLETED" in line:
                        # 找到任务名称和时间
                        if match:= re.search(completed_pattern1, line):
                            task_name = match.group(2)
                            time_value = float(match.group(4))
                        elif match := re.search(completed_pattern2, line):
                            task_name = match.group(2)
                            time_value = float(match.group(4))
                        if match:
                            if task_name not in completed_dict: 
                                completed_dict[task_name] = [time_value]
                            else:
                                completed_dict[task_name].append(time_value)
                    elif "MISSED DEADLINE" in line:
                        # 找到任务名称和时间
                        if match:= re.search(miss_pattern, line):
                            task_name = match.group(2)
                            time_value = float(match.group(4))

                        if match:
                            if task_name not in miss_dict:
                                miss_dict[task_name] = [time_value]
                            else:
                                miss_dict[task_name].append(time_value)
                    elif "triggered" in line:
                        if match:= re.search(trigger_pattern, line):
                            task_name = match.group(1)
                            time_value = float(match.group(2))
                        if match:
                            if task_name not in trigger_dict:
                                trigger_dict[task_name] = [time_value]
                            else:
                                trigger_dict[task_name].append(time_value)

            # 添加到列表中
            if filename not in force_list:
                completed_list.append(completed_dict)
                miss_list.append(miss_dict)
                filename_list.append(filename)
                trigger_list.append(trigger_dict)

    # find the common trigger event
    # glb_dyn_${x}_ideal.log.txt
    # glb_dyn_${x}_jitter_dis.log.txt
    # bin_pack_new_$x.log.txt
    # dyn_${x}_jitter_dis.log.txt

    # glb_dyn_${x}_jitter_en_seed_$seed.log.txt
    # dyn_${x}_jitter_en_seed_$seed.log.txt

    jitter_dis_index = []
    jitter_en_index = []
    groups = {}
    for i in range(len(trigger_list)):
        # if match:=re.match(jitter_en_log_fn_pattern, filename_list[i]):
        #     seed_id = int(match.group(3))
        if match:=re.match(log_pattern, filename_list[i]):
            info = get_group_dict(log_pattern_keys, match, log_pattern_type, True)
            if info['jitter_en'] and info['seed'] != '':
                seed_id = info['seed']
                groups[seed_id] = groups.get(seed_id, []) + [i]
            else:
                print("Unexpected log file name: ", filename_list[i])
        else:
            groups["static"] = groups.get("static", []) + [i]

    for group in groups.values():
        # print group
        print("=====================================")
        print("group:", [filename_list[idx] for idx in group])
        task_name_set = set()
        for idx in group:
            trigger_dict = trigger_list[idx]
            task_name_set.update(trigger_dict.keys())

        for task_name in task_name_set:
            trigger_event_set = set()
            # collect the trigger list from the group by task_name
            set_t = [set(trigger_list[idx][task_name]) for idx in group if task_name in trigger_list[idx]]
            if len(set_t) > 0:
                # find the common trigger event
                cm_trigger_event_set = set.intersection(*set_t)
                if verbose:
                    print(f"task: {task_name}, common element: {cm_trigger_event_set}\n")
                for idx in group:
                    if task_name in trigger_list[idx]:
                        unique_event_set = set(trigger_list[idx][task_name]) - cm_trigger_event_set
                        if len(unique_event_set) > 0:
                            print("task:", task_name)
                            print(f"{filename_list[idx]} unique element:", unique_event_set, "\n")
    df = pd.DataFrame()

    # 将列表转换为 DataFrame
    # columns by column extention
    for i in range(len(completed_list)):
        completed_dict = {key: len(value) for key, value in completed_list[i].items()}
        miss_dict = {key: len(value) for key, value in miss_list[i].items()}
        # 输出结果
        df_completed = pd.DataFrame.from_dict(completed_dict, orient="index", columns=["Completed Count"])
        df_missed = pd.DataFrame.from_dict(miss_dict, orient="index", columns=["Missed Count"])
        df = pd.concat([df, df_completed, df_missed], axis=1)
    # set multiindex
    df.columns = pd.MultiIndex.from_product([filename_list, ["Completed Count", "Missed Count"]])

    # add sum row
    df.loc["sum"] = df.sum(axis=0)
    df = df.fillna(0).astype(int)

    if get_ref_num_exec:
        num_exec = extract_num_exec(profiling_filename, aux_scale_factor, n_p, warmup_dis)
        # add num_exec row, and set it to the last row
        # add num_exc for every column
        # 创建一个包含 num_exec 值的 Series，索引与 df.columns 相同
        num_exec_series = pd.Series([num_exec] * len(df.columns), index=df.columns)
        
        # 将 num_exec_series 添加为 DataFrame 的最后一行
        df.loc["num_exec"] = num_exec_series

    # save to csv
    df.to_csv(output, index=True, header=True, encoding="utf-8-sig")


def test_mode():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", type=str, default="./log", help="path to log folder")
    argparser.add_argument("--output", type=str, default="./log/analyze.csv", help="path to output csv file")
    argparser.add_argument("--profiling_filename", type=str, default="profiling/profiling_light.csv", help="path to task profiling file")
    argparser.add_argument("--aux_scale_factor", type=float, default=1, help="auxiliary scaling factor")
    argparser.add_argument("--n_p", type=int, default=1, help="number of processors")
    argparser.add_argument("--get_ref_num_exec", action="store_true", help="get the reference number of execution")
    argparser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")
    argparser.add_argument("--sim_param_seq", type=str, default='sen,slowdown', help="sequence of simulation parameters")
    args = argparser.parse_args()
    print(f"===========folder: {args.folder}===========")
    sim_param_seq = args.sim_param_seq.split(",")
    log_pattern, log_pattern_keys, log_pattern_type = get_log_regexp(sim_param_seq) 
    count_miss_comp(
        args.folder, args.output, 
        log_pattern, log_pattern_keys, log_pattern_type,                     
        args.get_ref_num_exec,
        args.profiling_filename, args.aux_scale_factor, args.n_p, args.warmup_dis
    )

def get_scaner_warap(args):
    sim_param_seq = args.sim_param_seq.split(",")
    log_pattern, log_pattern_keys, log_pattern_type = get_log_regexp(sim_param_seq) 
    def scaner_warp(df, folder, info_dict):
        output = os.path.join(folder, args.stat_csv_filename)
        count_miss_comp(
            folder, output, 
            log_pattern, log_pattern_keys, log_pattern_type, 
            True, 
            args.profiling_filename, info_dict["aux_scale_factor"], args.n_p, args.warmup_dis, 
        )
    return scaner_warp

def scan_mode():    
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling_light.csv", help="profiling filename")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="throughput", help="filename")
    parser.add_argument("--stat_csv_filename", type=str, default="new_bin_pack.csv", help="csv filename")
    parser.add_argument("--folder_search_seq", type=str, default="num_cores,cfg_n", help="sequence of compile-time parameters")
    parser.add_argument("--n_p", type=int, default=3, help="number of processors")
    parser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")
    parser.add_argument("--sim_param_seq", type=str, default='sen,slowdown', help="sequence of simulation parameters")

    args = parser.parse_args()
    root_dir = args.root_dir
    search_seq = args.folder_search_seq.split(",")

    # Load the dataframe
    df = None
    root_path = os.path.join('log', root_dir)
    action_fn = get_scaner_warap(args)
    scanner = get_path_var_scaner([action_fn,], folder_pattern, folder_pattern_keys, folder_type, search_seq)
    scanner(df, root_path, len(search_seq), dict(), 0)

if __name__ == "__main__":
    # python -m analyze.stat_num_exec --folder /home/zhangchg/git_repo/scheduler/log/hist/before_asplos24summer/barycenter/core_scan/x3_0.09s_rda-80.00%\(T\)_5.00%\(S\)_all_soft/285 --output new_bin_pack.csv --n_p 3 --aux_scale_factor 3 --get_ref_num_exec
    # test_mode()
    # python -m analyze.stat_num_exec --root_dir hist/before_asplos24summer/barycenter/core_scan --folder_search_seq cfg_n,num_cores
    # python -m analyze.stat_num_exec --root_dir hist/before_asplos24summer/barycenter/aux_scan --folder_search_seq num_cores,cfg_n
    scan_mode()
    