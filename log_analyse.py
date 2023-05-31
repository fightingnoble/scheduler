import os
import argparse
import pandas as pd
import re

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder", type=str, default="./log", help="path to log folder")
argparser.add_argument("--output", type=str, default="./log/analyze.csv", help="path to output csv file")
args = argparser.parse_args()
folder = args.folder # log 文件夹路径

completed_list = []  # 用于存储已完成任务及其计数
miss_list = []  # 用于存储未完成任务及其计数
filename_list = []  # 用于存储文件名
trigger_list = []

# 遍历指定文件夹下的所有 .log.txt 文件, 按照文件名排序
# for filename in os.listdir(folder):
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".log.txt"):
        completed_dict = {}  # 用于存储已完成任务及其时间
        miss_dict = {}  # 用于存储未完成任务及其时间
        trigger_dict = {}
        with open(os.path.join(folder, filename), "r") as file:
            for line in file:
                if "COMPLETED" in line:
                    # 找到任务名称和时间
                    task_start = line.find(":") + 1
                    task_end = line.rfind("(")
                    task_name = line[task_start:task_end]
                    time_start = line.rfind("@") + 1
                    time_end = line.rfind("/")
                    time_value = float(line[time_start:time_end])
                    # 添加到 completed_dict 中
                    if task_name not in completed_dict: 
                        completed_dict[task_name] = [time_value]
                    else:
                        completed_dict[task_name].append(time_value)
                elif "MISSED DEADLINE" in line:
                    # 找到任务名称和时间
                    task_start = line.find(":") + 1
                    task_end = line.rfind("(")
                    task_name = line[task_start:task_end]
                    time_start = line.rfind("@") + 1
                    time_end = line.rfind("/")
                    time_value = float(line[time_start:time_end])
                    # 添加到 miss_dict 中
                    if task_name not in miss_dict:
                        miss_dict[task_name] = [time_value]
                    else:
                        miss_dict[task_name].append(time_value)
                elif "triggered" in line:
                    pattern = r'(?P<task>\w+)\s+triggered\s+@\s+(?P<time>\d+\.\d+)'

                    match = re.search(pattern, line)
                    if match:
                        task_name = match.group('task')
                        trigger_time = float(match.group('time'))
                        if task_name not in trigger_dict:
                            trigger_dict[task_name] = [trigger_time]
                        else:
                            trigger_dict[task_name].append(trigger_time)

        # 添加到列表中
        completed_list.append(completed_dict)
        miss_list.append(miss_dict)
        filename_list.append(filename)
        trigger_list.append(trigger_dict)

# find the common trigger event
# "bin_pack_new_256.log.txt", 
# "dyn_256_jitter_dis.log.txt", "dyn_256_jitter_en.log.txt", 
# "glb_dyn_256_jitter_dis.log.txt", "glb_dyn_256_jitter_en.log.txt", 

jitter_dis_index = []
jitter_en_index = []
for i in range(len(trigger_list)):
    if filename_list[i].endswith("jitter_dis.log.txt"):
        jitter_dis_index.append(i)
    elif filename_list[i].endswith("jitter_en.log.txt"):
        jitter_en_index.append(i)
    else:
        jitter_dis_index.append(i)

for group in [jitter_dis_index, jitter_en_index]:
    # print group
    print("=====================================")
    print("group:", [filename_list[idx] for idx in group])
    task_name_set = set()
    for idx in group:
        trigger_dict = trigger_list[idx]
        task_name_set.update(trigger_dict.keys())

    for task_name in task_name_set:
        print("task:", task_name)
        trigger_event_set = set()
        # find the common event
        set_t = [set(trigger_list[idx][task_name]) for idx in group if task_name in trigger_list[idx]]
        if len(set_t) > 0:
            cm_trigger_event_set = set.intersection(*set_t)
            for idx in group:
                if task_name in trigger_list[idx]:
                    unique_event_set = set(trigger_list[idx][task_name]) - cm_trigger_event_set
                    if len(unique_event_set) > 0:
                        print(f"{filename_list[idx]} unique element:", unique_event_set)
            print("common element:", cm_trigger_event_set)
        print()

# extract number of cores
def extract_num_cores(filename):
    pattern = r'(?P<num_cores>\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group('num_cores'))
    else:
        return -1

# num_cores = extract_num_cores(filename_list[0])

# import pickle

# # 加载 event_iter_dict_bin_pack_new_False.pkl 文件
# with open('event_iter_dict_bin_pack_new_False.pkl', 'rb') as f:
#     event_iter_dict_bin_pack_new_False = pickle.load(f)

# # 加载 event_iter_dict_dynamic_False.pkl 文件
# with open('event_iter_dict_dynamic_False.pkl', 'rb') as f:
#     event_iter_dict_dynamic_False = pickle.load(f)

# # 加载 event_iter_dict_dynamic_True.pkl 文件
# with open('event_iter_dict_dynamic_True.pkl', 'rb') as f:
#     event_iter_dict_dynamic_True = pickle.load(f)

# # 加载 event_iter_dict_glb_dynamic_False.pkl 文件
# with open('event_iter_dict_glb_dynamic_False.pkl', 'rb') as f:
#     event_iter_dict_glb_dynamic_False = pickle.load(f)

# # 加载 event_iter_dict_glb_dynamic_True.pkl 文件
# with open('event_iter_dict_glb_dynamic_True.pkl', 'rb') as f:
#     event_iter_dict_glb_dynamic_True = pickle.load(f)

# # 存储到词典中
# event_iter_dict = {
#     f"bin_pack_new_{num_cores}.log.txt": event_iter_dict_bin_pack_new_False,
#     f"dyn_{num_cores}_jitter_dis.log.txt": event_iter_dict_dynamic_False,
#     f"dyn_{num_cores}_jitter_en.log.txt": event_iter_dict_dynamic_True,
#     f"glb_dyn_{num_cores}_jitter_dis.log.txt": event_iter_dict_glb_dynamic_False,
#     f"glb_dyn_{num_cores}_jitter_en.log.txt": event_iter_dict_glb_dynamic_True
# }

# # filename_list, trigger_list
# actual_trigger_dict = dict(zip(filename_list, trigger_list))

# # 比较两个词典的值是否相等
# print("=================planned====================")
# a, b = event_iter_dict[f"dyn_{num_cores}_jitter_en.log.txt"], event_iter_dict[f"glb_dyn_{num_cores}_jitter_en.log.txt"]
# print({k:(a[k],b[k])for k in a if a[k] != b[k]})
# a, b = event_iter_dict[f"dyn_{num_cores}_jitter_dis.log.txt"], event_iter_dict[f"glb_dyn_{num_cores}_jitter_dis.log.txt"]
# print({k:(a[k],b[k])for k in a if a[k] != b[k]})
# print("=================actual====================")
# a, b = actual_trigger_dict[f"dyn_{num_cores}_jitter_en.log.txt"], actual_trigger_dict[f"glb_dyn_{num_cores}_jitter_en.log.txt"]
# print({k:(a[k],b[k])for k in a if a[k] != b[k]})
# a, b = actual_trigger_dict[f"dyn_{num_cores}_jitter_dis.log.txt"], actual_trigger_dict[f"glb_dyn_{num_cores}_jitter_dis.log.txt"]
# print({k:(a[k],b[k])for k in a if a[k] != b[k]})

df = pd.DataFrame()

# 将列表转换为 DataFrame
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

def merge_cells(group):
    group = group.applymap(str)
    group = group.replace({'nan': ''}) # 特殊处理空值
    
    # 计算每列列表示的长度
    col_widths = group.apply(max, axis=0).apply(len).to_list()
    
    # 将每列按照最长字符串的长度进行格式化，使每列长度一致
    fmt_str = '\n'.join(['{{:<{width}}}'.format(width=width) for width in col_widths])
    
    # 应用格式化字符串并合并每行
    result = group.apply(lambda x: fmt_str.format(*x.to_list()), axis=1)
    return result


grouped = df.groupby(level=0, axis=1)
merged_df = grouped.apply(merge_cells)

df.to_csv(args.output, index=True, header=True, encoding="utf-8-sig")