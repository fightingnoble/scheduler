import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="scalable_motiv", help="filename")
parser.add_argument("--file_o", type=str, default="scalable_motiv", help="filename")
parser.add_argument("--root_dir_glb", type=str, default="log", help="output directory")
parser.add_argument("--root_dir_dyn", type=str, default="log", help="output directory")
args = parser.parse_args()

# [90, 95, 99, 99.9, 99.99]
tgt_confidence = 0.99
tgt_confidence_idx = 3

df = pd.read_csv(os.path.join(args.root_dir_dyn, args.filename + ".csv"))
# Filter for lateness_mode == 'ignore'
aval_idx = (df['method'] == 'glb_dyn') & \
    (df['lateness_mode'] == 'all_soft') & \
    (df['num_cores'] == 400) & \
    (df['e2e_latency'] == 0.1) & \
    (df['jitter_en'] == True) 

avl_df = pd.DataFrame(df.loc[aval_idx])
# add ref line
avl_df.loc[aval_idx, "DDL_ref"] = avl_df["e2e_latency"]# * (1 - avl_df["temporal_rda_ratio"]/100)
avl_df.loc[aval_idx, "RT_ref"] = 0.1# * (1 - avl_df["temporal_rda_ratio"]/100)


# select the percentile columns
miss_rate = avl_df['miss_rate'].apply(lambda x: x.split(','))
miss_rate = miss_rate.apply(lambda x: [float(i) for i in x])
avl_df.loc[:,['mean']] = pd.DataFrame(miss_rate.tolist(), index=miss_rate.index).agg(['mean'], axis=1)

rt_percentile = avl_df['rt_percentile'].apply(lambda x: x.split(',')[tgt_confidence_idx])
ddl_percentile = avl_df['ddl_percentile'].apply(lambda x: x.split(',')[tgt_confidence_idx]).apply(lambda x: float(x))
avl_df['rt_percentile'] = rt_percentile.astype(float)
avl_df['ddl_percentile'] = ddl_percentile.astype(float)

# sort by aux_scale_factor
output = avl_df[['aux_scale_factor', 'rt_percentile', 'ddl_percentile', 'RT_ref', 'DDL_ref', "mean"]]
output.sort_values(by=['aux_scale_factor'], inplace=True)

print(output)
# Write to Excel
file_o = os.path.join(args.root_dir_dyn, args.file_o + ".xlsx")
writer = pd.ExcelWriter(file_o, engine='openpyxl')

output.to_excel(writer, sheet_name='Min Cores')

worksheet = writer.sheets['Min Cores']

# Add formatting, charts, etc

# Close writer
writer.close()