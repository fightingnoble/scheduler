import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="min_core", help="filename")
parser.add_argument("--root_dir_glb", type=str, default="log", help="output directory")
parser.add_argument("--root_dir_dyn", type=str, default="log", help="output directory")
args = parser.parse_args()

def extract_min_cores(args, root_dir, method='glb_dyn'):
    assert method in ['glb_dyn', 'dyn']
    df = pd.read_csv(os.path.join(root_dir, args.filename + ".csv"))
    # Filter for lateness_mode == 'ignore'
    aval_idx = (df['lateness_mode'] == 'ignore') & \
        (df['aux_scale_factor'] <= df['throughput']) & \
        (df['jitter_en'] == 'en') 

    avl_df = df.loc[aval_idx]
    tgt_df = avl_df.loc[avl_df['method'] == method]

    # Groupby scale factor and latency, find min cores 
    tgt_min_cores_idx = tgt_df.groupby(['e2e_latency', 'aux_scale_factor'])['num_cores'].idxmin()

    tgt_min_cores = tgt_df.loc[tgt_min_cores_idx]

    # extend the columns @ n_ctx_switch, cum_time
    columns_filter = ['e2e_latency', 'aux_scale_factor', 'num_cores', 'cum_time']
    tgt_min_cores = tgt_min_cores[columns_filter]
    tgt_min_cores["cum_time"] /= 0.4

    # set the index
    tgt_min_cores.set_index(['e2e_latency', 'aux_scale_factor'], inplace=True)
    return tgt_min_cores

glb_min_cores = extract_min_cores(args, root_dir=args.root_dir_glb, method='glb_dyn')
dyn_min_cores = extract_min_cores(args, root_dir=args.root_dir_dyn, method='dyn')

# get the common index
min_cores_idx = glb_min_cores.index.intersection(dyn_min_cores.index)
# filter the common index
glb_min_cores_comm = glb_min_cores.loc[min_cores_idx]
dyn_min_cores_comm = dyn_min_cores.loc[min_cores_idx]

# merge the two dataframes
min_cores = pd.merge(glb_min_cores_comm, dyn_min_cores_comm, on=['e2e_latency', 'aux_scale_factor'], suffixes=('_glb', '_dyn'))

print(min_cores)
file_o = os.path.join(args.root_dir_dyn, args.filename + ".xlsx")
writer = pd.ExcelWriter(file_o, engine='xlsxwriter')
workbook = writer.book 

min_cores.to_excel(writer, sheet_name='Min Cores')

worksheet = writer.sheets['Min Cores'] 

worksheet.set_column('A:F', 15)

writer.save()