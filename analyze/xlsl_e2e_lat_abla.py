# scanned parameters: 
# 1. bin 2. rationB 3. tp 
# 4. cfg: ratioA, ratioB, wsc, n_var, jitter_comp

# common: lat case seed bin*2, cfg *2 


# scan cfg
# sel: tp==9, bin in [4,8]
# list_jitter_comp_cfg=(0.2 0.2 0.15 0.1 0.05 0.15 0.1 0.05)
# list_exec_t_comp_ratioA=(0.3 0 0.25 0.2 0.15 0 0 0)

import pandas as pd
import argparse, os
from analyze.pattern import case_name_glb, case_name_dyn, case_name_cyc

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="min_core", help="filename")
parser.add_argument("--root_dir_glb", type=str, default="log", help="output directory")
parser.add_argument("--root_dir_dyn", type=str, default="log", help="output directory")
parser.add_argument("--outputfile", type=str, default="lat_ctx_abla", help="output file")
args = parser.parse_args()

def scan_size(args, root_dir):
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    
    # replace all 'nan' with ''
    df.fillna('', inplace=True)
    filter_idx = (df['lateness_mode'] == 'ignore') & \
               (df['aux_scale_factor'] == 9) & \
               (df['throughput'] >= df['aux_scale_factor']) & \
               (df['e2e_latency'].isin([0.09, 0.1, 0.08])) & \
               (df['wsc_slack_ratio'] == 100) & \
               (df['jitter_en'] == '') & \
                (df[['method', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'repack_ratio']].apply(tuple, axis=1).isin([('bin_pack_new', 20, 30, ''), ('bin_pack_new', 20, 0, '')])) 

    reduced_df = df.loc[filter_idx]
    # sort by num_bins
    reduced_df.sort_values(by=['num_bins'], inplace=True)
    # merge 'e2e_latency', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA' as tuple
    reduced_df['e2e_S(T)_S(J)'] = list(zip(reduced_df['e2e_latency'], reduced_df['jitter_t_comp_ratio'], reduced_df['exec_t_comp_ratioA']))
    
    # only pick 'e2e_S(T)_S(J)', 'num_bins', 'num_cores' and drop other columns
    reduced_df = reduced_df.drop(['lateness_mode', 'aux_scale_factor', 'throughput', 'wsc_slack_ratio', 'jitter_en', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'method','repack_ratio'], axis=1)
    df_cum_t = reduced_df.pivot_table(values=['num_cores'], index=['num_bins'], columns=['e2e_S(T)_S(J)'], aggfunc='first')
    df_cum_t.sort_index(axis=1, inplace=True)
    
    reduced_df = df_cum_t
    # save to file
    file_o = os.path.join(root_dir, "num_cores_scan_bin" + ".csv")
    reduced_df.to_csv(file_o, index=True)
    return reduced_df


def scan_bin(args, root_dir):
    # PGLB/DYN: Number of Bins v.s. number of context switches /normalized cumulated switches lateness
    # num_bins aux_scale_factor e2e_latency jitter_t_comp_ratio wsc_slack_ratio exec_t_comp_ratioA lateness_mode method jitter_en var_sen var_slowdown seed repack_ratio n_ctx_switch cum_time throughput num_cores
    # filter:
    # aux_scale_factor 9, and throughput >= aux_scale_factor
    # e2e_latency [0.09, 0.08, 0.1]
    # (method, jitter_t_comp_ratio,exec_t_comp_ratioA, repack_ratio) [(dyn, 20, 30, 0.15), (dyn, 20, 0, 0.05), (pglb, 20, 30, ''), (pglb, 20, 0, '')]
    # wsc_slack_ratio 100
    # lateness_mode ignore
    # method [dyn, pglb]
    # jitter_en en
    
    # n_ctx_switch cum_time are the values of interest
    # other dims are the index 

    # action: 
    # 1. group by method, e2e_latency, (jitter_t_comp_ratio,exec_t_comp_ratioA, repack_ratio), num_bins
    # 2. in each group, reduce n_ctx_switch, cum_time along seed by mean, max, min
    # 3. get the ralation between n_ctx_switch, cum_time w.r.t. num_bins
    # 4. divide the cum_time by 0.4 to get the normalized cumulated switches lateness
    # 5. Get the error bar: replace min column with mean-min, max column with max-mean

        
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    
    # Apply all filters
    # [0.09, 0.08, 0.1]
    # ('dyn', 20, 30, 0.15), ('dyn', 20, 0, 0.05), ('pglb', 20, 30, float('nan')), ('pglb', 20, 0, float('nan'))
    # set nan to ''
    df['repack_ratio'] = df['repack_ratio'].fillna('')
    filter_idx = (df['lateness_mode'] == 'ignore') & \
               (df['aux_scale_factor'] == 9) & \
               (df['throughput'] >= df['aux_scale_factor']) & \
               (df['e2e_latency'].isin([0.09,])) & \
               (df['wsc_slack_ratio'] == 100) & \
               (df['jitter_en'] == 'en') & \
                (df[['method', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'repack_ratio']].apply(tuple, axis=1).isin([('dyn', 20, 30, 0.15), ('pglb', 20, 30, '')])) 
               
    filtered_df = df.loc[filter_idx]

    # Groupby method, e2e_latency, jitter_t_comp_ratio, exec_t_comp_ratioA
    grouped = filtered_df.groupby(['method', 'e2e_latency', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'num_bins', 'repack_ratio'])

    # Reduce n_ctx_switch, cum_time along seed by mean, max, min
    reduced_df = grouped.agg({
        'n_ctx_switch': ['mean', 'max', 'min'],
        'cum_time': ['mean', 'max', 'min']
    }).reset_index()

    # Get the relation between n_ctx_switch, cum_time w.r.t. num_bins
    reduced_df['norm_cum_time'] = reduced_df['cum_time']['mean'] / 0.4

    # Get the error bar: replace min column with mean-min, max column with max-mean
    reduced_df[('cum_time', 'min')] = reduced_df[('cum_time', 'mean')] - reduced_df[('cum_time', 'min')]
    reduced_df[('cum_time', 'max')] = reduced_df[('cum_time', 'max')] - reduced_df[('cum_time', 'mean')]
    reduced_df[('n_ctx_switch', 'min')] = reduced_df[('n_ctx_switch', 'mean')] - reduced_df[('n_ctx_switch', 'min')]
    reduced_df[('n_ctx_switch', 'max')] = reduced_df[('n_ctx_switch', 'max')] - reduced_df[('n_ctx_switch', 'mean')]

    # sort by num_bins
    reduced_df.sort_values(by=['num_bins'], inplace=True)
    # flattern multi-index
    reduced_df.columns = ['_'.join(col).strip() if col[-1] else col[0] for col in reduced_df.columns.values]
    # merge 'e2e_latency', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA' as tuple
    reduced_df['e2e_S(T)_S(J)'] = list(zip(reduced_df['e2e_latency'], reduced_df['jitter_t_comp_ratio'], reduced_df['exec_t_comp_ratioA']))
    
    reduced_df = reduced_df[['method', 'e2e_S(T)_S(J)', 'num_bins', 'norm_cum_time', 'n_ctx_switch_mean', 'n_ctx_switch_max', 'n_ctx_switch_min', 'cum_time_mean', 'cum_time_max', 'cum_time_min']]
    # 'num_bins' as index, 'norm_cum_time', as columns, stacked by 'e2e_S(T)_S(J)' and 'method' horizontally
    df_cum_t = reduced_df.pivot_table(values=['norm_cum_time', 'n_ctx_switch_mean'], index=['num_bins'], columns=['e2e_S(T)_S(J)','method'], aggfunc='first')
    # df_cum_t.columns = reduced_df.columns.swaplevel(0, 1)
    df_cum_t.sort_index(axis=1, inplace=True)
    
    # # 'n_ctx_switch_mean' 
    # df_num = reduced_df.pivot_table(values=['n_ctx_switch_mean'], index=['num_bins'], columns=['e2e_S(T)_S(J)','method'], aggfunc='first')
    # # df_num.columns = reduced_df.columns.swaplevel(0, 1)
    # df_num.sort_index(axis=1, inplace=True)
    
    # # put them together
    # reduced_df = pd.concat([df_cum_t, df_num], axis=1)
    reduced_df = df_cum_t
    # save to file
    file_o = os.path.join(root_dir, "ctx_scan_bin" + ".csv")
    reduced_df.to_csv(file_o, index=True)
    
    return reduced_df

def scan_tp(args, root_dir):
        
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    
    # replace all 'nan' with ''
    df.fillna('', inplace=True)

    filter_idx = (df['lateness_mode'] == 'ignore') & \
               (df['num_bins'].isin([4, 8])) & \
               (df['throughput'] >= df['aux_scale_factor']) & \
               (df['e2e_latency'].isin([0.09,])) & \
               (df['wsc_slack_ratio'] == 100) & \
               (df['jitter_en'] == 'en') & \
                (df[['method', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'repack_ratio']].apply(tuple, axis=1).isin([('dyn', 20, 30, 0.15), ('pglb', 20, 30, '')])) 
               
    filtered_df = df.loc[filter_idx]

    # Groupby method, e2e_latency, jitter_t_comp_ratio, exec_t_comp_ratioA
    grouped = filtered_df.groupby(['method', 'e2e_latency', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'repack_ratio', 'num_bins', 'aux_scale_factor'])

    # Reduce n_ctx_switch, cum_time along seed by mean, max, min
    reduced_df = grouped.agg({
        'n_ctx_switch': ['mean', 'max', 'min'],
        'cum_time': ['mean', 'max', 'min']
    }).reset_index()

    # Get the relation between n_ctx_switch, cum_time w.r.t. num_bins
    reduced_df['norm_cum_time'] = reduced_df['cum_time']['mean'] / 0.4

    # Get the error bar: replace min column with mean-min, max column with max-mean
    reduced_df[('cum_time', 'min')] = reduced_df[('cum_time', 'mean')] - reduced_df[('cum_time', 'min')]
    reduced_df[('cum_time', 'max')] = reduced_df[('cum_time', 'max')] - reduced_df[('cum_time', 'mean')]
    reduced_df[('n_ctx_switch', 'min')] = reduced_df[('n_ctx_switch', 'mean')] - reduced_df[('n_ctx_switch', 'min')]
    reduced_df[('n_ctx_switch', 'max')] = reduced_df[('n_ctx_switch', 'max')] - reduced_df[('n_ctx_switch', 'mean')]

    # sort by num_bins
    reduced_df.sort_values(by=['num_bins'], inplace=True)
    # flattern multi-index
    reduced_df.columns = ['_'.join(col).strip() if col[-1] else col[0] for col in reduced_df.columns.values]
    # merge 'e2e_latency', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA' as tuple
    reduced_df['e2e_S(T)_S(J)'] = list(zip(reduced_df['e2e_latency'], reduced_df['jitter_t_comp_ratio'], reduced_df['exec_t_comp_ratioA']))
    
    reduced_df = reduced_df[['method', 'e2e_S(T)_S(J)', 'num_bins', 'aux_scale_factor', 'norm_cum_time', 'n_ctx_switch_mean', 'n_ctx_switch_max', 'n_ctx_switch_min', 'cum_time_mean', 'cum_time_max', 'cum_time_min']]
    df_cum_t = reduced_df.pivot_table(values=['norm_cum_time', 'n_ctx_switch_mean'], index=['aux_scale_factor'], columns=['num_bins', 'method','e2e_S(T)_S(J)',], aggfunc='first')
    reduced_df = df_cum_t
    # save to file
    file_o = os.path.join(root_dir, "scan_tp" + ".csv")
    reduced_df.to_csv(file_o, index=True)    
    return reduced_df


def scan_ratioB(args, root_dir):
        
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    
    # replace all 'nan' with ''
    df.fillna('', inplace=True)

    filter_idx = (df['lateness_mode'] == 'ignore') & \
               (df['num_bins'].isin([4, 8])) & \
               (df['aux_scale_factor'] == 9) & \
               (df['throughput'] >= df['aux_scale_factor']) & \
               (df['e2e_latency'].isin([0.09, 0.1, 0.08])) & \
               (df['wsc_slack_ratio'] == 100) & \
               (df['jitter_en'] == 'en') & \
                (df[['method', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA']].apply(tuple, axis=1).isin([('dyn', 20, 0,),])) 
               
    filtered_df = df.loc[filter_idx]

    # Groupby method, e2e_latency, jitter_t_comp_ratio, exec_t_comp_ratioA
    grouped = filtered_df.groupby(['e2e_latency', 'num_bins', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA', 'repack_ratio'])

    # Reduce n_ctx_switch, cum_time along seed by mean, max, min
    reduced_df = grouped.agg({
        'n_ctx_switch': ['mean', 'max', 'min'],
        'cum_time': ['mean', 'max', 'min']
    }).reset_index()

    # Get the relation between n_ctx_switch, cum_time w.r.t. num_bins
    reduced_df['norm_cum_time'] = reduced_df['cum_time']['mean'] / 0.4

    # Get the error bar: replace min column with mean-min, max column with max-mean
    reduced_df[('cum_time', 'min')] = reduced_df[('cum_time', 'mean')] - reduced_df[('cum_time', 'min')]
    reduced_df[('cum_time', 'max')] = reduced_df[('cum_time', 'max')] - reduced_df[('cum_time', 'mean')]
    reduced_df[('n_ctx_switch', 'min')] = reduced_df[('n_ctx_switch', 'mean')] - reduced_df[('n_ctx_switch', 'min')]
    reduced_df[('n_ctx_switch', 'max')] = reduced_df[('n_ctx_switch', 'max')] - reduced_df[('n_ctx_switch', 'mean')]

    # sort by num_bins
    reduced_df.sort_values(by=['num_bins'], inplace=True)
    # flattern multi-index
    reduced_df.columns = ['_'.join(col).strip() if col[-1] else col[0] for col in reduced_df.columns.values]
    # merge 'e2e_latency', 'jitter_t_comp_ratio', 'exec_t_comp_ratioA' as tuple
    reduced_df['e2e_S(T)_S(J)'] = list(zip(reduced_df['e2e_latency'], reduced_df['jitter_t_comp_ratio'], reduced_df['exec_t_comp_ratioA']))
    
    reduced_df = reduced_df[['e2e_S(T)_S(J)', 'num_bins', 'repack_ratio', 'norm_cum_time', 'n_ctx_switch_mean', 'n_ctx_switch_max', 'n_ctx_switch_min', 'cum_time_mean', 'cum_time_max', 'cum_time_min']]
    df_cum_t = reduced_df.pivot_table(values=['norm_cum_time', 'n_ctx_switch_mean'], index=['repack_ratio'], columns=['num_bins','e2e_S(T)_S(J)',], aggfunc='first')
    reduced_df = df_cum_t
    # save to file
    file_o = os.path.join(root_dir, "scan_ratioB" + ".csv")
    reduced_df.to_csv(file_o, index=True)    
    return reduced_df


# scan_bin(args, args.root_dir_glb) 
# scan_size(args, args.root_dir_dyn) 
# scan_tp(args, args.root_dir_dyn) 
scan_ratioB(args, args.root_dir_dyn)