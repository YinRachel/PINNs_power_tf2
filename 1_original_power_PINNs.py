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
    def __init__(self, model, X_u, X_f, u_train, nu):
        self.model = model
        
        self.X_u_data = X_u
        self.X_f_data = X_f
        self.u_train = u_train
        
        self.t_u = X_u[:,1:2]
        self.x_u = X_u[:,0:1]
        self.t_f = X_f[:,1:2]
        self.x_f = X_f[:,0:1]
        
        self.nu = nu
        
    def net_u(self,X_u_data):
        return self.model(X_u_data) 

    def net_f(self,X_f_train):
        with tf.GradientTape(persistent=True) as tape:
        
            tape.watch(X_f_train)
            u = self.model(X_f_train)
            gradients = tape.gradient(u, X_f_train)
            u_t = gradients[:, 1]
            
        u_tt = tape.gradient(u_t,X_f_train)[:, 1]
        x = X_f_train[:,0]
        z = self.nu *tf.math.sin(u)
        z = tf.squeeze(z)
        f = 0.4 * u_tt +0.15 *u_t + z - x

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

    
nu=0.2;
noise = 0.0        

# a set of Nu=40 randomly distributed initial and boundary data across the entire spatiotemporal domain, 
# Nf= 8′000 collocation points, and a 5-layer neural network with 10 neurons per hidden layer.

N_u = 50
N_f = 8000 
# 第一个数字表示输入，中间表示层中的神经元数量，最后一个数字决定了要预测几个变量
# The first number represents the input, the middle represents the number of neurons in the layer, 
# and the last number determines how many variables are to be predicted.
layers = [2, 10, 10, 10, 10, 10, 1]

data = scipy.io.loadmat('./data/data_1_try.mat')

# 从加载的数据中提取时间数据t，
# 然后使用flatten()将其变为一维数组，并通过[:,None]增加一个维度，使其成为列向量。
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
print("t",t.shape)
print("x",x.shape)
# 提取物理系统的真实响应，这里假设它存储在usol键下。
# 使用np.real确保响应是实数。.T是转置操作，以符合后续处理的需要。
Exact = np.real(data['usol']).T

# 使用np.meshgrid函数生成空间位置x和时间t的网格。这个网格将用于定义在物理域内的点，以便于在这些点上进行物理模拟或神经网络预测。
X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              
X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
# Doman bounds

lb=np.array([0.08 ,  0.        ])
ub=np.array([0.18,  20.        ])

# 准备训练数据点和对应的物理响应:
xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
xx2 = np.hstack((X[:,0:1], T[:,0:1]))
xx3 = np.hstack((X[:,-1:], T[:,-1:]))

X_u_train = np.vstack([xx1, xx2, xx3])

#  使用拉丁超立方抽样(LHS)生成 X_f_train
# Generate X_f_train using Latin Hypercube Sampling (LHS)
X_f_train = lb + (ub-lb)*lhs(2, N_f)

# 将 X_f_train 与 X_u_train 合并,目的是确保网络训练时既考虑到了物理方程在整个域内的约束，也考虑到了特定的已知条件（X_u_train）
X_f_train = np.vstack((X_f_train, X_u_train))

# 这行代码通过垂直堆叠（vstack）合并了三个数据集uu1、uu2和uu3来形成u_train数据集。
# 这些数据集uu1、uu2和uu3可能代表了在物理系统的不同边界或初始条件下的已知物理响应。
# 合并这些数据集后，u_train包含了所有用于训练的物理响应值。
uu1 = Exact[0:1,:].T
uu2 = Exact[:,0:1]
uu3 = Exact[:,-1:]
u_train = np.vstack([uu1, uu2, uu3])

# 这行代码生成了一个随机索引数组idx。np.random.choice函数从X_u_train的第一个维度（即数据点的总数）中随机选择N_u个索引，
# 而且replace=False确保选择的索引是唯一的，即不会有重复的索引，这样每个选中的数据点都是独一无二的。
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)

# 使用idx索引数组来更新X_u_train数据集，
# 这意味着从原始的X_u_train数据集中选取了N_u个随机的数据点，作为新的X_u_train数据集用于训练。
X_u_train = X_u_train[idx, :]
u_train = u_train[idx,:]

X_u_train = tf.convert_to_tensor(X_u_train, dtype=tf.float32)
u_train = tf.convert_to_tensor(u_train, dtype=tf.float32)
X_f_train = tf.convert_to_tensor(X_f_train, dtype=tf.float32)

model = PhysicsInformedNN(layers=layers, lb=lb, ub=ub)
solver = PINNs_Slover(model, X_u_train, X_f_train, u_train, nu)
solver.fit(1000,X_u_train, X_f_train, u_train)

start_time = time.time()
u_pred, f_pred = solver.predict(X_star)
elapsed = time.time() - start_time              

# print('Training time: %.4f' % (elapsed))
# print("u_pred",u_pred)

# print("u_star",u_star)
# print("X_star",X_star)

# # 展平数据
u_star_flattened = u_star.flatten()
u_pred_flattened = u_pred.numpy().flatten()

# # 打印长度和数组内容的前几个值
# print("u_star_flattened 长度:", len(u_star_flattened))
# print("u_pred_flattened 长度:", len(u_pred_flattened))
# print("u_star_flattened 前几个值:", u_star_flattened[:10])
# print("u_pred_flattened 前几个值:", u_pred_flattened[:10])
l2_error = np.sqrt(np.sum((u_pred_flattened - u_star_flattened)**2))
print("L2 Error:", l2_error)
# 确保时间数组的长度与展平后的数据长度一致
time = np.linspace(0, 20, len(u_star_flattened))

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(time, u_star_flattened, label="real value (u_star)", linestyle='-')
plt.plot(time, u_pred_flattened, label="predict value (u_pred)", linestyle='--')

plt.xlabel('time')
plt.ylabel('u(rad)')
plt.legend()
plt.title('the comparison between real and predict')
plt.show()