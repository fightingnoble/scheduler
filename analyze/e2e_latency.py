import pandas as pd
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="timing", help="filename")
parser.add_argument("--root_dir_glb", type=str, default="log", help="output directory")
parser.add_argument("--root_dir_dyn", type=str, default="log", help="output directory")
args = parser.parse_args()

def extract_glb(args):
    df = pd.read_csv(os.path.join(args.root_dir_glb, args.filename + ".csv"))
    # Filter for lateness_mode == 'ignore'
    aval_idx = (df['method'] == 'glb_dyn') & \
        (df['confidence'] == 99) 

    avl_df = df.loc[aval_idx]
    glb_df_soft = avl_df.loc[(df['jitter_en'] == True) & (avl_df['lateness_mode'] == 'all_soft')]
    glb_df_ignore = avl_df.loc[(df['jitter_en'] == True) & (avl_df['lateness_mode'] == 'ignore')]

    # set the index
    glb_df_soft.set_index(['e2e_latency', 'aux_scale_factor', 'num_cores'], inplace=True)
    glb_df_ignore.set_index(['e2e_latency', 'aux_scale_factor','num_cores'], inplace=True)

    # select coloumns ddl_percentile	rt_percentile for glb_df_soft and glb_df_ignore
    glb_df_soft = glb_df_soft[['ddl_percentile', 'rt_percentile', 'miss_rate']]
    glb_df_ignore = glb_df_ignore[['ddl_percentile', 'rt_percentile', 'miss_rate']]
    return glb_df_soft,glb_df_ignore

def extract_dyn(args):
    df = pd.read_csv(os.path.join(args.root_dir_dyn, args.filename + ".csv"))
    # Filter for lateness_mode == 'ignore'
    aval_idx = (df['confidence'] == 99)  & \
        (df['method'] == 'dynamic') & \
        (df['lateness_mode'] == 'ignore') 

    avl_df = df.loc[aval_idx]
    # add ref line
    avl_df.loc[:, "DDL_ref"] = avl_df["e2e_latency"]# * (1 - avl_df["temporal_rda_ratio"]/100)
    avl_df.loc[:, "RT_ref"] = 0.1# * (1 - avl_df["temporal_rda_ratio"]/100)

    dyn_df = avl_df.loc[(avl_df['jitter_en'] == True)]
    static_df = avl_df.loc[(avl_df['jitter_en'] != True)]

    # set the index
    dyn_df.set_index(['e2e_latency', 'aux_scale_factor', 'num_cores'], inplace=True)
    static_df.set_index(['e2e_latency', 'aux_scale_factor', 'num_cores'], inplace=True)

    # select coloumns ddl_percentile	rt_percentile for glb_df_soft and glb_df_ignore
    static_df = static_df[['ddl_percentile', 'rt_percentile', 'miss_rate']]
    dyn_df = dyn_df[['ddl_percentile', 'rt_percentile', 'miss_rate', 'DDL_ref', 'RT_ref']]
    return dyn_df,static_df

dyn_df, static_df = extract_dyn(args)
glb_df_soft, glb_df_ignore = extract_glb(args)

# get the common index
min_cores_idx = glb_df_soft.index.intersection(glb_df_ignore.index).intersection(dyn_df.index).intersection(static_df.index)
# filter the common index
glb_df_soft_comm = glb_df_soft.loc[min_cores_idx]
glb_df_ignore_comm = glb_df_ignore.loc[min_cores_idx]
dyn_df_comm = dyn_df.loc[min_cores_idx]
static_df_comm = static_df.loc[min_cores_idx]

# merge the all dataframes and add multicolumns
merged_df = pd.concat([glb_df_soft_comm[['ddl_percentile', 'rt_percentile']], 
                       glb_df_ignore_comm[['ddl_percentile', 'rt_percentile']], 
                       static_df_comm[['ddl_percentile', 'rt_percentile']],
                       dyn_df_comm[['ddl_percentile', 'rt_percentile']], 
                       dyn_df_comm[['DDL_ref', 'RT_ref']]], 
                       axis=1, keys=['glb_soft', 'glb_ignore', 'dyn_s', 'dyn', 'ref'])

# get miss_rate column from glb_df_soft_comm, glb_df_ignore_comm, dyn_df_comm, static_df_comm
# Extract miss_rate columns
miss_soft = glb_df_soft_comm['miss_rate']
miss_ignore = glb_df_ignore_comm['miss_rate'] 
miss_dyn = dyn_df_comm['miss_rate']
miss_static = static_df_comm['miss_rate']

# # Append columns to current DataFrame
miss_df = pd.concat([miss_soft, miss_ignore, miss_static, miss_dyn], axis=1, keys=['glb_soft', 'glb_ignore', 'dyn_s', 'dyn'])
# add a hierarchical index
miss_df.columns = pd.MultiIndex.from_product([['miss_rate'], miss_df.columns])
# merge the miss_df with merged_df
merged_df = pd.concat([merged_df, miss_df], axis=1)

# sort the index
merged_df.sort_index(inplace=True)

print(merged_df)
file_o = os.path.join(args.root_dir_dyn, args.filename + ".xlsx")
writer = pd.ExcelWriter(file_o, engine='xlsxwriter')

# Group by columns
grouped = merged_df.groupby(['e2e_latency', 'aux_scale_factor']) 
for name, group in grouped:
    scale, latency = name
    group.to_excel(writer, sheet_name=f'{scale}_{latency}')
    # worksheet = writer.sheets['timing'] 
    # worksheet.set_column('A:H', 15)
writer.save()