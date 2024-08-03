import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument('--log_file', type=str, help='path to log file')
args = argparser.parse_args()

# 创建空列表来存储提取的数据
data = {
    'case': [],
    'var_sen': [],
    'var_slowdown': [],
    'flag': []
}
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

title_dict = {
    'pglb': "Cyc-RT (S)", 
    'cyclic': "Cyc-RT"
}

axis_dict = {
    "var_sen": r"$\sigma_{sen}$", 
    "var_slowdown": r"$\sigma_{sl}$"
}

# 定义正则表达式模式来匹配日志中的数据
pattern = r'(\w+)\svar_([\d.]+)\(J\)_([\d.]+)\(T\) (\w+)\.'

# 打开包含日志的文件
with open(args.log_file, 'r') as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            # print(line)
            data['case'].append(match.group(1))
            data['var_sen'].append(float(match.group(2)))
            data['var_slowdown'].append(1/(1-float(match.group(3))))
            data['flag'].append(match.group(4))

# 创建DataFrame
df = pd.DataFrame(data)
# sort by case, var_sen, var_slowdown
df = df.sort_values(by=['case', 'var_sen', 'var_slowdown'])
# save to csv file to the parent directory of log_file
df.to_csv(args.log_file.replace('txt', 'csv'), index=False)

# 绘图
font_title = {'size': 18, 'weight': 'bold'}
font_label = {'size': 18}
cases = df['case'].unique()
for casel in cases:
    subset = df[df['case'] == casel]
    plt.figure()
    for flag, color in [('success', 'green'), ('fail', 'red')]:
        subset_flag = subset[subset['flag'] == flag]
        plt.scatter(subset_flag['var_slowdown'], subset_flag['var_sen'], c=color, label=flag)
    
    # 找出边界点
    success_boundary = []
    fail_boundary = []
    for var_slowdown in df['var_slowdown'].unique():
        subset_slowdown = df[(df['var_slowdown'] == var_slowdown) & (df['case'] == casel)]
        if not subset_slowdown.empty:
            max_sen_success = subset_slowdown['var_sen'][subset_slowdown['flag'] == 'success'].max()
            if not pd.isna(max_sen_success):
                # print(var_slowdown, subset_slowdown)
                success_boundary.append((var_slowdown, max_sen_success))
            
            min_sen_fail = subset_slowdown['var_sen'][subset_slowdown['flag'] == 'fail'].min()
            fail_boundary.append((var_slowdown, min_sen_fail))    

    # 补全边界点
    # def ensure_points_around_boundary(boundary, df):
    #     var_slowdown, var_sen = boundary
    #     nearby_points = df[
    #         ((df['var_slowdown'] == var_slowdown - 0.01) | (df['var_slowdown'] == var_slowdown) | (df['var_slowdown'] == var_slowdown + 0.01)) &
    #         ((df['var_sen'] == var_sen - 0.01) | (df['var_sen'] == var_sen) | (df['var_sen'] == var_sen + 0.01))
    #     ]
    #     missing_points = []
    #     for slow_diff in [-0.01, 0, 0.01]:
    #         for sen_diff in [-0.01, 0, 0.01]:
    #             if not nearby_points[(nearby_points['var_slowdown'] == var_slowdown + slow_diff) & (nearby_points['var_sen'] == var_sen + sen_diff)].any().any():
                    
                    
    #                 missing_points.append((var_slowdown + slow_diff, var_sen + sen_diff))
    #     return missing_points

    # missing_points = []
    # for boundary in success_boundary + fail_boundary:
    #     if not pd.isna(boundary[1]):
    #         missing_points.extend(ensure_points_around_boundary(boundary, df))

    # # 将补全的点加入DataFrame
    # for point in missing_points:
    #     df = df.append({'case': 'generated', 'var_slowdown': point[0], 'var_sen': point[1], 'flag': 'unknown'}, ignore_index=True)

    # 绘制边界
    if success_boundary:
        success_boundary = sorted(success_boundary, key=lambda x: (x[0], x[1]))
        if success_boundary[-1][1] > 0: 
            success_boundary.append((success_boundary[-1][0], 0))
        boundary_x, boundary_y = zip(*success_boundary)
        plt.plot(boundary_x, boundary_y, 'go-', label='Success Boundary')
    # print(success_boundary)
    # var_slowdown == 0.35 and var_sen == 0.2
    # print(df[(df['var_slowdown'] == 0.35) & (df['var_sen'] == 0.2)])
    
    if fail_boundary:
        fail_boundary = sorted(fail_boundary, key=lambda x: (x[0], x[1]))
        boundary_x, boundary_y = zip(*fail_boundary)
        plt.plot(boundary_x, boundary_y, 'ro--', label='Fail Boundary')

    plt.xlabel(f'{axis_dict["var_slowdown"]}', fontdict=font_label)
    plt.ylabel(f'{axis_dict["var_sen"]}', fontdict=font_label)
    plt.title(f'Case: {title_dict[casel]}', fontdict=font_title)
    plt.legend()
    plt.savefig(args.log_file.replace('txt', f'{casel}.pdf').replace('log', 'plot'))
    plt.close()
