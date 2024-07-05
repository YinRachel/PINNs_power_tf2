import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time
from pyDOE import lhs
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

np.random.seed(1234)
tf.random.set_seed(1234)


nu=0.5
noise = 0.0        
P1m = 0.18
N_u = 100
N_f = 10000 
Epoch = 5000
layers = [2, 15, 15, 15, 15, 15, 1]

data = scipy.io.loadmat('./data/mat_data/gen_1_data.mat')

t = data['t'].flatten()[:,None]
x1 = data['x1'].flatten()[:,None]

t_min = np.min(t)
t_max = np.max(t)

# 计算位置x1的最小值和最大值
x1_min = np.min(x1)
x1_max = np.max(x1)

Exact = np.real(data['usol1'])

lb = np.array([t_min, x1_min])
ub = np.array([t_max, x1_max])

X_star = np.hstack((t, x1))
u_star = Exact.flatten()[:, None]

total_points = N_u -2
indices = np.linspace(0, len(X_star) - 1, total_points, dtype=int)

# 提取对应的X和u值
X_u_train = X_star[indices, :]
u_train = u_star[indices, :]

#  使用拉丁超立方抽样(LHS)生成 X_f_train
# Generate X_f_train using Latin Hypercube Sampling (LHS)
X_f_train = lb + (ub-lb)*lhs(2, N_f)

# 将 X_f_train 与 X_u_train 合并,目的是确保网络训练时既考虑到了物理方程在整个域内的约束，也考虑到了特定的已知条件（X_u_train）
X_f_train = np.vstack((X_f_train, X_u_train))


X_u_train = tf.convert_to_tensor(X_u_train, dtype=tf.float32)
u_train = tf.convert_to_tensor(u_train, dtype=tf.float32)
X_f_train = tf.convert_to_tensor(X_f_train, dtype=tf.float32)
X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)

# model = PhysicsInformedNN(layers=layers, lb=lb, ub=ub)
# solver = PINNs_Slover(model,u_train, nu, P1m)
# start_time_train = time.time()
# solver.fit(Epoch,X_u_train, X_f_train, u_train)
# train_time = time.time() - start_time_train
# print('training time: %.4f' % (train_time))

# start_time = time.time()
# u_pred, f_pred = solver.predict(X_star)
# elapsed = time.time() - start_time              

# print('Training time: %.4f' % (elapsed))

# # # 展平数据
# u_star_flattened = u_star.flatten()
# u_pred_flattened = u_pred.numpy().flatten()

# l2_error = np.sqrt(np.sum((u_pred_flattened[50:] - u_star_flattened[50:])**2))
# print("L2 Error:", l2_error)

# # # 确保时间数组的长度与展平后的数据长度一致
# time = np.linspace(t_min, t_max, len(u_star_flattened))[50:]

# # # 绘图
# plt.figure(figsize=(10, 5))
# plt.plot(time, u_star_flattened[50:], label="real value", linestyle='-')
# plt.plot(time, u_pred_flattened[50:], label="predict value", linestyle='--')

# plt.xlabel('time')
# plt.ylabel('u(rad)')
# plt.legend()
# plt.title('Generator 1')
# plt.show()
# #plt.savefig("network_graph.png")

# u_pred_df = pd.DataFrame(u_pred.numpy(),columns=['u_pred'])
# u_pred_df.to_excel('./results/gen_1.xlsx')