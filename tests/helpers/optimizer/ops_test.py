import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torch

# 设置初始参数值
init_params = {
    'min_val': 0.5,
    'ceil_factor': 5.0,
    'comp_scale': 30.0,
    'eq_scale': 30.0,
    'y_val': 1.0, 
    'sp_factor': 10.0,
}

# 创建图形和子图布局
fig = plt.figure(figsize=(15, 15))
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, hspace=0.4)

# 创建函数计算辅助函数
def softplus(x, beta=1):
    return torch.log(1 + torch.exp(beta*x))/beta

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# 创建7个子图
axes = [fig.add_subplot(4, 2, i+1) for i in range(7)]
plot_lines = []
original_lines = []

# 生成数据点
x = torch.linspace(-2, 5, 701).round(decimals=2)

# 函数1: clamp_min_diff
def func1(x, min_val, sp_factor):
    return softplus(x - min_val, sp_factor) + min_val

# 函数2: minimum_diff
def func2(x, y_val, sp_factor):
    return x - softplus(x - y_val, sp_factor)

# 函数3: maximum_diff
def func3(x, y_val, sp_factor):
    return x + softplus(y_val - x, sp_factor)

# 函数4: ceil_diff
def func4(x, ceil_factor):
    fractional = x - torch.floor(x)
    return x + (1 - sigmoid(ceil_factor * (fractional - 0.5)))

# 函数5: ge_diff
def func5(x, y_val, comp_scale):
    return sigmoid(comp_scale * (x - y_val))

# 函数6: le_diff
def func6(x, y_val, comp_scale):
    return sigmoid(comp_scale * (y_val - x))

# 函数7: eq_diff
def func7(x, y_val, eq_scale):
    return torch.exp(-eq_scale * (x - y_val)**2)

# 初始化所有图表
titles = [
    "clamp_min_diff = softplus(x - min_val) + min_val",
    "minimum_diff = x - softplus(x - y)",
    "maximum_diff = x + softplus(y - x)",
    "ceil_diff = x + (1 - sigmoid(factor*(frac-0.5)))",
    "ge_diff = sigmoid(scale*(x - y))",
    "le_diff = 1 - sigmoid(scale*(y - x))",
    "eq_diff = exp(-scale*(x - y)^2)"
]

for i, ax in enumerate(axes):
    # 绘制可逆函数
    if i == 0:
        y = func1(x, init_params['min_val'], init_params['sp_factor'])
    elif i in [1, 2]:
        y = [func2, func3][i-1](x, init_params['y_val'], init_params['sp_factor'])
    elif i == 3:
        y = func4(x, init_params['ceil_factor'])
    elif i == 4:
        y = func5(x, init_params['y_val'], init_params['comp_scale'])
    elif i == 5:
        y = func6(x, init_params['y_val'], init_params['comp_scale'])
    elif i == 6:
        y = func7(x, init_params['y_val'], init_params['eq_scale'])
    
    line, = ax.plot(x.numpy(), y.numpy(), 'b-', linewidth=2)
    plot_lines.append(line)
    
    # 绘制原始函数
    if i == 0:
        orig_y = torch.maximum(x, torch.tensor(init_params['min_val']))
    elif i == 1:
        orig_y = torch.minimum(x, torch.tensor(init_params['y_val']))
    elif i == 2:
        orig_y = torch.maximum(x, torch.tensor(init_params['y_val']))
    elif i == 3:
        orig_y = torch.ceil(x)
    elif i == 4:
        orig_y = (x >= init_params['y_val']).float()
    elif i == 5:
        orig_y = (x <= init_params['y_val']).float()
    elif i == 6:
        orig_y = (x == init_params['y_val']).float()
    
    if i!= 6:
        orig_line, = ax.plot(x.numpy(), orig_y.numpy(), 'r--', linewidth=1.5)
    else:
        orig_line = ax.scatter(x.numpy(), orig_y.numpy(), c='r', marker='o', s=50)
    original_lines.append(orig_line)
    
    ax.set_title(titles[i])
    ax.grid(True)
    ax.set_xlim(-2, 5)
    
    if i in [3, 4, 5, 6]:
        ax.set_ylim(-0.1, 1.5)
    else:
        ax.set_ylim(-2, 5)


# 添加滑块
slider_ax3 = fig.add_axes([0.1, 0.02, 0.2, 0.03])
ceil_factor_slider = Slider(slider_ax3, 'Ceil Factor', 1.0, 50.0, valinit=init_params['ceil_factor'])

slider_ax4 = fig.add_axes([0.1, 0.06, 0.2, 0.03])
comp_scale_slider = Slider(slider_ax4, 'Comp Scale', 1.0, 50.0, valinit=init_params['comp_scale'])

slider_ax = fig.add_axes([0.35, 0.02, 0.2, 0.03])
min_val_slider = Slider(slider_ax, 'Min Value', -1.0, 3.0, valinit=init_params['min_val'])

slider_ax2 = fig.add_axes([0.35, 0.06, 0.2, 0.03])
y_val_slider = Slider(slider_ax2, 'Y Value', -1.0, 3.0, valinit=init_params['y_val'])

slider_ax5 = fig.add_axes([0.6, 0.02, 0.2, 0.03])
eq_scale_slider = Slider(slider_ax5, 'Eq Scale', 1.0, 50.0, valinit=init_params['eq_scale'])

slided_ax1 = fig.add_axes([0.6, 0.06, 0.2, 0.03])
sp_factor_slider = Slider(slided_ax1, 'SP Factor', 1.0, 50.0, valinit=init_params['sp_factor'])

# 更新函数
def update(val):
    min_val = min_val_slider.val
    y_val = y_val_slider.val
    sp_factor = sp_factor_slider.val
    ceil_factor = ceil_factor_slider.val
    comp_scale = comp_scale_slider.val
    eq_scale = eq_scale_slider.val
    
    # 更新所有图表
    plot_lines[0].set_ydata(func1(x, min_val, sp_factor).numpy())
    original_lines[0].set_ydata(torch.maximum(x, torch.tensor(min_val)).numpy())
    
    plot_lines[1].set_ydata(func2(x, y_val, sp_factor).numpy())
    original_lines[1].set_ydata(torch.minimum(x, torch.tensor(y_val)).numpy())
    
    plot_lines[2].set_ydata(func3(x, y_val, sp_factor).numpy())
    original_lines[2].set_ydata(torch.maximum(x, torch.tensor(y_val)).numpy())
    
    plot_lines[3].set_ydata(func4(x, ceil_factor).numpy())
    original_lines[3].set_ydata(torch.ceil(x).numpy())
    
    plot_lines[4].set_ydata(func5(x, y_val, comp_scale).numpy())
    original_lines[4].set_ydata((x >= y_val).float().numpy())
    
    plot_lines[5].set_ydata(func6(x, y_val, comp_scale).numpy())
    original_lines[5].set_ydata((x <= y_val).float().numpy())
    
    plot_lines[6].set_ydata(func7(x, y_val, eq_scale).numpy())
    original_lines[6].set_ydata((x == y_val).float().numpy())
    
    fig.canvas.draw_idle()

# 注册更新函数
min_val_slider.on_changed(update)
y_val_slider.on_changed(update)
sp_factor_slider.on_changed(update)
ceil_factor_slider.on_changed(update)
comp_scale_slider.on_changed(update)
eq_scale_slider.on_changed(update)

# 添加重置按钮
reset_ax = fig.add_axes([0.8, 0.06, 0.1, 0.04])
reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')

def reset(event):
    min_val_slider.reset()
    y_val_slider.reset()
    sp_factor_slider.reset()
    ceil_factor_slider.reset()
    comp_scale_slider.reset()
    eq_scale_slider.reset()

reset_button.on_clicked(reset)

plt.show()