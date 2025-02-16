import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

# 假设数据
y1 = np.array([0.57814, 0.4262025, 0.2634925, 0.10115, 0.08497, 0.082125, 0.07155, 0.044705, 0.0395475, 0.0313975])
y2= np.array([0.27088, 0.122085, 0.0342525, 0.0159225, 0.0162775, 0.0123425, 0.01288, 0.0066175, 0.003925, 0.0017825,] )
y = y2
x= 590/np.array([1, 2, 4, 8, 10, 12, 16, 18, 20, 24] )

# 定义二次函数模型
def fit_func(x, a):
    return a * x**2
# def fit_func(x, a, b):
#     return a * x**2 + b * x**1
def fit_func(x, a, b):
    return a * x**3 + b * x**2



# 使用 curve_fit 拟合数据
popt, pcov = curve_fit(fit_func, x, y)

# 获取拟合的参数（a, b, c）
print(f"拟合参数: {dict(zip(['a', 'b', 'c', 'd'], popt))}")

# 获取参数的标准误差（标准差）
perr = np.sqrt(np.diag(pcov))
print(f"参数的标准误差: {dict(zip(['a', 'b', 'c', 'd'], perr))}")

# 绘制数据和拟合曲线
x_fit = np.linspace(min(x), max(x), 100)
y_fit = fit_func(x_fit, *popt)

# 计算 t 值和 p 值
t_values = popt / perr  # t 值 = 参数估计值 / 标准误差
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=len(x) - len(popt))) for t in t_values]  # p 值，df = len(x) - 2（自由度）


# 计算 R^2
# 1. 计算总平方和 TSS
tss = np.sum((y - np.mean(y))**2)
# 2. 计算残差平方和 RSS
rss = np.sum((y - fit_func(x, *popt))**2)
# 3. 计算 R^2
r2 = 1 - rss / tss

print(f"R^2: {r2:.4f}")

print(f"t 值: {t_values}")
print(f"p 值: {p_values}")

plt.scatter(x, y, label='Data', color='blue')
plt.plot(x_fit, y_fit, label='Fitted curve', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
