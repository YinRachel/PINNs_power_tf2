# https://ieeexplore.ieee.org/document/9282004
# https://github.com/gmisy/Physics-Informed-Neural-Networks-for-Power-Systems/tree/master


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

# self
# X_u 由时间和空间点组成的输入数据集，通常代表的是物理系统的已知条件。
# u 相位角 （真实值），要求解的值
# X_f 用于在整个定义域内施加物理定律的点集(确保数据满足物理定律)
# layers = [2, 10, 10, 10, 10, 10, 1]
# lb lower bound [0.08,0.18]
# ub upper bound [0.,20.]
# nu = 0.2 B12

# 
# nu=0.2;
# noise = 0.0        

# N_u = 40
# N_f = 8000 表示想要生成的点的数量

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, layers,lb, ub, activation='tanh', kernel_initializer='glorot_normal',**kwargs):
        super().__init__(**kwargs)
        # layers = [2, 10, 10, 10, 10, 10, 1]
        self.num_hidden_layers = len(layers) - 2
        self.output_dim = layers[-1] 
        self.lb = lb
        self.ub = ub
        
        
        self.hidden = [tf.keras.layers.Dense(units,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for units in layers[1:-1]]
        self.out = tf.keras.layers.Dense(layers[-1])
    
    def call(self, inputs):
        X = 2.0 * (inputs - self.lb) / (self.ub-self.lb) - 1.0
        for layers in self.hidden:
            X = layers(X)
        return self.out(X)
    
class PINNs_Slover():
    def __init__(self, model,u_train, nu, P1m):
        self.model = model
        self.P1m = P1m
        self.nu = nu
        self.u_train = u_train
        
    def net_u(self,X_u_data):
        return self.model(X_u_data) 

    def net_f(self,X_f_train):
        with tf.GradientTape(persistent=True) as tape:
        
            tape.watch(X_f_train)
            u = self.model(X_f_train)
            gradients = tape.gradient(u, X_f_train)
            u_t = gradients[:, 0]
            
        u_tt = tape.gradient(u_t,X_f_train)[:, 0]
        x1 = X_f_train[:,1]
        z = self.nu *tf.math.sin(u - x1)
        z = tf.squeeze(z)
        f = 0.4 * u_tt +0.15 *u_t + z - self.P1m

        return f

    def loss_fn(self, u_true, u_pred, f_pred):
        data_loss = tf.reduce_mean(tf.square(u_true - u_pred))
        physics_loss = tf.reduce_mean(tf.square(f_pred))
        return data_loss + physics_loss
    
    def train_step(self,X_u_train, X_f_train):
        with tf.GradientTape(persistent=True) as tape:
            
            u_pred = self.net_u(X_u_train)
            f_pred = self.net_f(X_f_train)
            
            loss = self.loss_fn(self.u_train, u_pred, f_pred)
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def fit(self, epochs, X_u_train, X_f_train, u_train, optimizer=tf.optimizers.Adam()):
        self.optimizer = optimizer
        
        for epoch in range(epochs):
            loss = self.train_step(X_u_train, X_f_train,)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss:{loss.numpy()}")
    
    def predict(self, X_star):
        u_star = self.net_u(X_star)
        f_star = self.net_f(X_star)
        return u_star,f_star


nu=0.5
noise = 0.0        
P1m = 0.18
N_u = 100
N_f = 10000 
Epoch = 5000
layers = [2, 15, 15, 15, 15, 15, 1]

data = scipy.io.loadmat('./data/mat_data/gen_2_data.mat')

t = data['t'].flatten()[:,None]
x2 = data['x2'].flatten()[:,None]

t_min = np.min(t)
t_max = np.max(t)

# 计算位置x1的最小值和最大值
x2_min = np.min(x2)
x2_max = np.max(x2)

Exact = np.real(data['usol2'])

lb = np.array([t_min, x2_min])
ub = np.array([t_max, x2_max])

X_star = np.hstack((t, x2))
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

model = PhysicsInformedNN(layers=layers, lb=lb, ub=ub)
solver = PINNs_Slover(model,u_train, nu, P1m)
solver.fit(Epoch,X_u_train, X_f_train, u_train)

start_time = time.time()
u_pred, f_pred = solver.predict(X_star)
elapsed = time.time() - start_time              

print('Training time: %.4f' % (elapsed))
# print("u_pred",u_pred)

# print("u_star",u_star)
# print("X_star",X_star)

# # 展平数据
u_star_flattened = u_star.flatten()
u_pred_flattened = u_pred.numpy().flatten()

l2_error = np.sqrt(np.sum((u_pred_flattened[50:] - u_star_flattened[50:])**2))
print("L2 Error:", l2_error)

# # 确保时间数组的长度与展平后的数据长度一致
time = np.linspace(t_min, t_max, len(u_star_flattened))[50:]

# # 绘图
plt.figure(figsize=(10, 5))
plt.plot(time, u_star_flattened[50:], label="real value", linestyle='-')
plt.plot(time, u_pred_flattened[50:], label="predict value", linestyle='--')

plt.xlabel('time')
plt.ylabel('u(rad)')
plt.legend()
plt.title('Generator 2')
plt.show()
#plt.savefig("network_graph.png")

u_pred_df = pd.DataFrame(u_pred.numpy(),columns=['u_pred'])
u_pred_df.to_excel('./results/gen_2.xlsx')