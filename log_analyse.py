import os
import argparse
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder", type=str, default="./log", help="path to log folder")
args = argparser.parse_args()
folder = args.folder # log 文件夹路径

completed_list = []  # 用于存储已完成任务及其计数
miss_list = []  # 用于存储未完成任务及其计数
filename_list = []  # 用于存储文件名

# 遍历指定文件夹下的所有 .log.txt 文件
for filename in os.listdir(folder):
    if filename.endswith(".log.txt"):
        completed_dict = {}  # 用于存储已完成任务及其时间
        miss_dict = {}  # 用于存储未完成任务及其时间
        with open(os.path.join(folder, filename), "r") as file:
            for line in file:
                if "COMPLETED" in line:
                    # 找到任务名称和时间
                    task_start = line.find(":") + 1
                    task_end = line.rfind("(")
                    task_name = line[task_start:task_end]
                    time_start = line.rfind("@") + 1
                    time_end = line.rfind("\n")
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
                    time_end = line.rfind("\n")
                    time_value = float(line[time_start:time_end])
                    # 添加到 miss_dict 中
                    if task_name not in miss_dict:
                        miss_dict[task_name] = [time_value]
                    else:
                        miss_dict[task_name].append(time_value)
        # 添加到列表中
        completed_list.append(completed_dict)
        miss_list.append(miss_dict)
        filename_list.append(filename)


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

df.to_csv("log/analyze.csv")