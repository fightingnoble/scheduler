import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="throughput", help="filename")
parser.add_argument("--root_dir_glb", type=str, default="log", help="output directory")
parser.add_argument("--root_dir_dyn", type=str, default="log", help="output directory")
args = parser.parse_args()

# filter columns: lateness_mode == 'ignore' 
def extract_max_tp(args, root_dir, method='glb_dyn', jitter_en='en'):
    assert method in ['glb_dyn', 'dyn']
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    # Filter for lateness_mode == 'ignore'
    aval_idx = (df['lateness_mode'] == 'ignore') & \
        (df['jitter_en'] == jitter_en) & \
        (df['method'] == method)
    
    # set index and sort by latency (descending) and num_cores (ascending)
    avl_df = df.loc[aval_idx, ['e2e_latency', 'num_cores', 'throughput']]
    avl_df.set_index(['e2e_latency', 'num_cores'], inplace=True)
    avl_df.sort_index(ascending=[False, True], inplace=True)
    return avl_df

max_tp_df_glb_en = extract_max_tp(args, args.root_dir_glb, method='glb_dyn', jitter_en='en')
max_tp_df_glb_dis = extract_max_tp(args, args.root_dir_glb, method='glb_dyn', jitter_en='dis')
max_tp_df_dyn_en = extract_max_tp(args, args.root_dir_dyn, method='dyn', jitter_en='en')
max_tp_df_dyn_dis = extract_max_tp(args, args.root_dir_dyn, method='dyn', jitter_en='dis')

# get common index
idx = max_tp_df_glb_en.index.intersection(max_tp_df_dyn_en.index)
max_tp_df_glb_en = max_tp_df_glb_en.loc[idx]
max_tp_df_glb_dis = max_tp_df_glb_dis.loc[idx]
max_tp_df_dyn_en = max_tp_df_dyn_en.loc[idx]
max_tp_df_dyn_dis = max_tp_df_dyn_dis.loc[idx]

# merge the dataframes
max_tp_df = pd.concat([max_tp_df_glb_en, max_tp_df_dyn_en, max_tp_df_glb_dis, max_tp_df_dyn_dis], axis=1)
max_tp_df.columns = ['Planaria (EN)', 'Ours (EN)', 'Planaria (DIS)', 'Ours (DIS)']


print(max_tp_df)

# Convert column dtypes

# Write to Excel
file_o = os.path.join(args.root_dir_dyn, args.filename + ".xlsx")
writer = pd.ExcelWriter(file_o, engine='openpyxl')

max_tp_df.to_excel(writer, sheet_name='Max Throughput')

worksheet = writer.sheets['Max Throughput']

# Add formatting, charts, etc

# Close writer
writer.close()
