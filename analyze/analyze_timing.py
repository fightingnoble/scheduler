import re
from analyze.pattern import *
from functools import reduce
import pandas as pd
import os
import numpy as np
from analyze.stat_num_exec import extract_num_exec
from model.message.Context_message import trace_analyser

from utils import update_df

folder_search_seq = ["cfg_n"]

def set_percetile(df, data, e2e_latency_list, miss_rate):
        
    rt_e2e_latency_list = np.array(e2e_latency_list[0])
    ddl_e2e_latency_list = np.array(e2e_latency_list[1])
    print(f"mean: {np.mean(rt_e2e_latency_list):.6f}, std: {np.std(rt_e2e_latency_list):.6f}, max: {np.max(rt_e2e_latency_list):.6f}, min: {np.min(rt_e2e_latency_list):.6f}")
    print(f"mean: {np.mean(ddl_e2e_latency_list):.6f}, std: {np.std(ddl_e2e_latency_list):.6f}, max: {np.max(ddl_e2e_latency_list):.6f}, min: {np.min(ddl_e2e_latency_list):.6f}")
            # calculate the percentile
    rt_percentiles = np.concatenate([np.percentile(rt_e2e_latency_list, [90, 95, 99, 99.9, 99.99]), rt_e2e_latency_list.max(keepdims=True)])
    ddl_percentiles = np.concatenate([np.percentile(ddl_e2e_latency_list, [90, 95, 99, 99.9, 99.99]), ddl_e2e_latency_list.max(keepdims=True)])
    print(f"rt_percentile: {rt_percentiles}")
    print(f"ddl_percentile: {ddl_percentiles}")

    # Append a row to the dataframe with the values
    df = update_df(df, data, {'ddl_percentile': ','.join([f'{x:.6f}' for x in ddl_percentiles.tolist()]),
                                'rt_percentile': ','.join([f'{x:.6f}' for x in rt_percentiles.tolist()]),
                                'confidence': ','.join([f'{x}' for x in [90, 95, 99, 99.9, 99.99]]),
                                'miss_rate': ','.join([f'{x:.3f}' for x in miss_rate]) if isinstance(miss_rate, list) else miss_rate})
    return df

def get_e2e_checker(file_pattern, file_pattern_keys, file_pattern_type, timing_flag_dict, profiling_filename=None, n_p=None, 
                    warmup_dis=None, stat_csv_filename=None, index_seq=None, trace_and_log_check_en=False):
    def e2e_checker(df, folder, info_dict):
        aux_scale_factor = info_dict['aux_scale_factor']
        num_exec, sink_pred = extract_num_exec(profiling_filename, aux_scale_factor, n_p, warmup_dis, "e2e")

        root,dirs,files = os.walk(folder).__next__()
        assert len(dirs) == 0
        dist_recoder = {}
        miss_rate_recoder = {}
        for fn in sorted(os.listdir(folder)):
            fn_match = re.match(file_pattern, fn)
            if fn_match:
                file_path = os.path.join(folder, fn)
                print(file_path)
                # Create a dictionary with the values
                data = {**info_dict, **get_group_dict(file_pattern_keys, fn_match, file_pattern_type)}
                data.pop("", None)
                if 'jitter_en' not in data:
                    data['jitter_en'] = False
                if 'seed' not in data:
                    data['seed'] = ""

                # false log output: the ones with jitter_en == True but seed == "" and the ones with jitter_en == False but seed != ""
                if data['jitter_en'] and data['seed'] == "" or data['jitter_en'] == False and data['seed'] != "":
                    continue
                
                # representation:
                # len(e2e_latency_list[0]) + len(e2e_latency_list[1])
                # n_exec
                # n_comp, n_miss

                # definitly
                # save trace if and only if job is completed
                # n_comp == len(e2e_latency_list[0]) + len(e2e_latency_list[1])

                # if all soft: all compelete:
                # n_comp == n_exec == len(e2e_latency_list[0]) + len(e2e_latency_list[1])
                # else
                # not all tasks are issued, num_exec is not necessary equal to n_comp+n_miss or len(e2e_latency_list[0]) + len(e2e_latency_list[1])

                if trace_and_log_check_en:
                    if stat_csv_filename is not None:
                        log_folder_t = folder.replace('trace', 'log').split('/')
                        log_folder = os.path.join(*log_folder_t[0:-1])
                        for idx in index_seq:
                            if idx == 'num_cores':
                                log_folder = os.path.join(log_folder, str(data['num_cores']))
                            else:
                                log_folder = os.path.join(log_folder, log_folder_t[-1])
                        root,dirs,files = os.walk(log_folder).__next__()
                        assert len(dirs) == 0

                        for log_fn in files:
                            match = re.match(stat_csv_filename, log_fn)
                            if match:
                                break
                        if not match:
                            print(f"!!! Warning: no stat file in {log_folder} !!!")
                            return df
                        log_path = os.path.join(log_folder, log_fn)
                        stat_df = pd.read_csv(log_path, index_col=0, header=[0, 1])
                        # r"(dyn|glb_dyn)(_\d+)?_jitter_(dis|en)\.log\.txt"
                        # index_name = 'dyn' if data['method'] == case_name_dyn else case_name_glb

                        # if args.test_case == case_name_pglb_input:
                        #     case_pth = case_name_pglb
                        # elif args.test_case == case_name_cyc_input:
                        #     case_pth = case_name_cyc
                        # elif args.test_case == case_name_dyn_input:
                        #     case_pth = case_name_dyn

                        if data['method'] == case_name_dyn:
                            case_name = case_name_dyn
                        elif data['method'] == case_name_glb:
                            case_name = case_name_glb
                        elif data['method'] == case_name_cyc:
                            case_name = case_name_cyc
                        elif data['method'] == case_name_pglb:
                            case_name = case_name_pglb

                        # !@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@!@
                        index_name += f"_{data['num_cores']}" 
                        index_name += '_jitter_dis' if not data['jitter_en'] else '_jitter_en'
                        index_name += f'_seed_{data["seed"]}' if data["seed"]!='' and data['jitter_en'] else ''
                        index_name += '.log.txt'
                        # @#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@#@
                        # case_name + jitter_en_${suffix[@]:4}_seed_$seed
                        # index_name = f"{case_name}_jitter_{'en' if data['jitter_en'] else 'dis'}"

                        if data['seed'] != '':
                            index_name += f"_seed_{data['seed']}"
                        n_comp = 0 
                        for idx in stat_df.loc[:, index_name]["Completed Count"].index:
                            if "_".join(idx.split('_')[0:-2]) in sink_pred:
                                n_comp += stat_df.loc[:, index_name]["Completed Count"][idx]
                    if data["lateness_mode"] == "all_soft":
                        n_miss = 0
                        for idx in stat_df.loc[:, index_name]["Missed Count"].index:
                            if "_".join(idx.split('_')[0:-2]) in sink_pred:
                                n_miss += stat_df.loc[:, index_name]["Missed Count"][idx] 
                        # assert n_miss == 0 

                if data["lateness_mode"] == "all_soft" and data['method'] == 'glb_dyn':
                    e2e_latency_list, n_violation = trace_analyser(timing_flag_dict, file_path, data["e2e_latency"], data["lateness_mode"], True)
                    # glb_dyn: missed jobs are also completed
                    # dyn: some jobs may failed to be placed
                    # bool checking is failed, check the miss rate by timing directly
                    if trace_and_log_check_en:
                        assert num_exec == len(e2e_latency_list[0]) + len(e2e_latency_list[1])
                    miss_rate = n_violation / num_exec
                else:
                    # missed jobs are not completed, and some tasks are even not issued (i.e., either not completed or missed)
                    e2e_latency_list = trace_analyser(timing_flag_dict, file_path, data["e2e_latency"], data["lateness_mode"])                        
                    miss_rate = 1- (len(e2e_latency_list[0]) + len(e2e_latency_list[1])) / num_exec

                if trace_and_log_check_en:
                    print(f"n_comp: {n_comp}, len_trace:{len(e2e_latency_list[0]) + len(e2e_latency_list[1])}")
                    try:
                        assert n_comp == len(e2e_latency_list[0]) + len(e2e_latency_list[1]) 
                    except:
                        print(f"20231126: CodingError, try to gurrante the trace log consistancy, but failed, n_comp: {n_comp}, len_trace:{len(e2e_latency_list[0]) + len(e2e_latency_list[1])}")
                                
                if data['jitter_en']:
                    # record the latency and miss rate data before histogram analysis
                    idx = (data['method'], data['num_cores'])
                    if idx not in dist_recoder:
                        dist_recoder[idx] = []
                    dist_recoder[idx].append(e2e_latency_list)
                    if idx not in miss_rate_recoder:
                        miss_rate_recoder[idx] = []
                    miss_rate_recoder[idx].append(miss_rate)
                else:
                    # aplly histogram analysis
                    data.pop('seed')
                    df = set_percetile(df, data, e2e_latency_list, miss_rate)
        
        # merge the list of each key
        for method, num_cores in dist_recoder.keys():
            data = {**info_dict, **{'method': method, 'num_cores': num_cores, 'jitter_en': True}}
            data.pop("", None)
            e2e_latency_list = reduce(lambda x, y: [x[0]+y[0], x[1]+y[1]], dist_recoder[(method, num_cores)])
            miss_rate = miss_rate_recoder[(method, num_cores)]
            df = set_percetile(df, data, e2e_latency_list, miss_rate)

        return df

    return e2e_checker

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="timing", help="filename")
    parser.add_argument("--folder_search_seq", type=str, default=','.join(folder_search_seq), help="core list")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling_light.csv", help="profiling filename")
    parser.add_argument("--stat_csv_filename", type=str, default=r"new_bin_pack(\d+)?.csv", help="csv filename")
    parser.add_argument("--n_p", type=int, default=3, help="number of processors")
    parser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")
    parser.add_argument("--index_seq", type=str, default='num_cores,cfg_n', help="core list")

    args = parser.parse_args()
    root_dir = args.root_dir
    
    # load the criticality 
    from task.task_cfg import load_taskattrib
    glb_n_task_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=False) 
    timing_flag_dict = {}
    for task_name in glb_n_task_dict:
        timing_flag_dict[task_name] = glb_n_task_dict[task_name].timing_flag

    filename = args.filename
    filename = f"{filename}.csv"
    filename = os.path.join(args.output_dir, filename)
    folder_search_seq = args.folder_search_seq.split(",")
    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=
            [key for search_key in folder_search_seq for key in folder_pattern_keys[search_key] if key != ""]
            # keys in trace pattern except seed, to acheive distribution statistics
            + ['method', "num_cores", "jitter_en"] 
            +['confidence', 'ddl_percentile', 'rt_percentile', 'miss_rate']).to_csv(filename, index=False)
    
    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('trace', root_dir)
    ctx_extracter = get_e2e_checker(trace_pattern, trace_pattern_keys, trace_pattern_type, timing_flag_dict, 
                                    args.profiling_filename, args.n_p, args.warmup_dis, args.stat_csv_filename, 
                                    args.index_seq.split(","))
    scanner = get_path_var_scaner([ctx_extracter, ], folder_pattern, folder_pattern_keys, folder_type, folder_search_seq)
    df = scanner(df, root_path, len(folder_search_seq), dict(), 0)
    df.to_csv(filename, index=False)
