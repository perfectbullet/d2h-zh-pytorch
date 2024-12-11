import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# 定义物体的初始状态
x0 = 0  # 初始位置（米）
v0 = 0  # 初始速度（米/秒）
a0 = 1  # 初始加速度（米/秒^2）
# 定义时间步长和总时间
dt = 0.01  # 时间步长（秒）
t_end = 10  # 总时间（秒）
# 定义物体的运动方程
def model(x, t):
    x_dot = x + a0 * dt  # 速度公式
    x_ddot = 0  # 加速度公式（这里假设加速度不变）
    return [x_dot, x_ddot]
# 初始化解向量
x = np.zeros((2,))
x[0] = x0  # 初始位置
x[1] = v0  # 初始速度
t = np.linspace(0, t_end, int(t_end / dt))  # 时间向量
# 求解常微分方程
y = odeint(model, x, t)
x_plot = y[:, 0]  # 提取位置数据并绘制图形
v_plot = y[:, 1]  # 提取速度数据并绘制图形
# 绘制结果图形
plt.plot(t, x_plot, label='Position (m)')
plt.plot(t, v_plot, label='Velocity (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.legend()
plt.show()