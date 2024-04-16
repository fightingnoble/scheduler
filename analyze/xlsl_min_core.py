import pandas as pd
import argparse, os
from analyze.pattern import case_name_glb, case_name_dyn, case_name_cyc

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="min_core", help="filename")
parser.add_argument("--root_dir_glb", type=str, default="log", help="output directory")
parser.add_argument("--root_dir_dyn", type=str, default="log", help="output directory")
args = parser.parse_args()

def extract_min_cores(args, root_dir, method=case_name_glb, jitter_en=True):
    assert method in [case_name_glb, case_name_dyn]
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    # Filter for lateness_mode == 'ignore'
    aval_idx = (df['lateness_mode'] == 'ignore') & \
        (df['aux_scale_factor'] <= df['throughput']) & \
        (df['jitter_en'] == jitter_en) 

    avl_df = df.loc[aval_idx]
    tgt_df = avl_df.loc[avl_df['method'] == method]

    # Groupby scale factor and latency, find min cores 
    tgt_min_cores = tgt_df.groupby(['e2e_latency', 'aux_scale_factor'])['num_cores'].min().reset_index()
    tgt_min_cores.set_index(['e2e_latency', 'aux_scale_factor'], inplace=True)
    for idx in tgt_min_cores.index:
        group = tgt_df.groupby(['e2e_latency', 'aux_scale_factor']).get_group(idx)
        group = group[group['num_cores'] == tgt_min_cores.loc[idx, 'num_cores']]

        tgt_min_cores.loc[idx, 'avg_cum_time'] = group['cum_time'].mean()
        tgt_min_cores.loc[idx, 'min_cum_time'] = group['cum_time'].min()
        tgt_min_cores.loc[idx, 'max_cum_time'] = group['cum_time'].max()
        
    min_cores = tgt_min_cores[['num_cores']]
    # tgt_min_cores["avg_cum_time"] /= 0.4
    # tgt_min_cores["min_cum_time"] /= 0.4
    # tgt_min_cores["max_cum_time"] /= 0.4
    ctx_switch_stat = tgt_min_cores.loc[:,["avg_cum_time", "min_cum_time", "max_cum_time"]]
    ctx_switch_stat["min_cum_time"] = (ctx_switch_stat["avg_cum_time"] - ctx_switch_stat["min_cum_time"])
    ctx_switch_stat["max_cum_time"] = (ctx_switch_stat["max_cum_time"] - ctx_switch_stat["avg_cum_time"])
    ctx_switch_stat /= 0.4

    return min_cores, ctx_switch_stat

glb_min_cores, glb_ctx_switch_stat = extract_min_cores(args, root_dir=args.root_dir_glb, method=case_name_glb)
dyn_min_cores, dyn_ctx_switch_stat = extract_min_cores(args, root_dir=args.root_dir_dyn, method=case_name_dyn)
glb_min_cores_static, glb_ctx_switch_stat_static = extract_min_cores(args, root_dir=args.root_dir_glb, method=case_name_glb, jitter_en=False)
dyn_min_cores_static, dyn_ctx_switch_stat_static = extract_min_cores(args, root_dir=args.root_dir_dyn, method=case_name_dyn, jitter_en=False)

# merge the dataframes
min_cores = pd.concat([glb_min_cores, glb_min_cores_static, dyn_min_cores, dyn_min_cores_static], axis=1)
min_cores.columns = pd.MultiIndex(levels=[['glb', 'glb_static', case_name_dyn, 'dyn_static'], ['min_cores']], codes=[[0, 1, 2, 3], [0, 0, 0, 0]], names=['method', 'stat'])
ctx_switch_stat = pd.concat([glb_ctx_switch_stat, glb_ctx_switch_stat_static, dyn_ctx_switch_stat, dyn_ctx_switch_stat_static], axis=1)
ctx_switch_stat.columns = pd.MultiIndex.from_product([['glb', 'glb_static', case_name_dyn, 'dyn_static'], ['avg_cum_time', 'min_cum_time', 'max_cum_time']], names=['method', 'stat'])
# set column names
output = pd.concat([min_cores, ctx_switch_stat], axis=1)

# set multi columns

# sort by latency(descending) and scale factor(ascending)
output.sort_index(level=[0, 1], ascending=[False, True], inplace=True)

print(output)

# Convert column dtypes

# Write to Excel
file_o = os.path.join(args.root_dir_dyn, args.filename + ".xlsx")
writer = pd.ExcelWriter(file_o, engine='openpyxl')

output.to_excel(writer, sheet_name='Min Cores')

worksheet = writer.sheets['Min Cores']

# Add formatting, charts, etc

# Close writer
writer.close()
