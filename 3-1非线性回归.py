import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点,U型二次函数曲线  np.newaxis使得有shape,1列
# np.linspace(-0.5,0.5,200).shape  == (200,) 没有shape
# x_data.shape == (200,1)
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder   None个样本，每个样本的特征为1，即第0层神经单元（样本的特征）为1个
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

num_L0, num_L1,num_L2 = [1,10,1] # num_L2是输出层了
# 定义神经网络中间层
# 10标识10个神经单元，1对应前面第0层神经单元个数，输出结束当然是[前面层样本量，本层神经单元数量]
Weights_L1 = tf.Variable(tf.random_normal([num_L0,num_L1]))
biases_L1 = tf.Variable(tf.zeros([1,num_L1]))
# [None,1]*[1,10]=[None,10]
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([num_L1,num_L2]))
biases_L2 = tf.Variable(tf.zeros([1,num_L2]))
# [None,num_L1]*[num_L1,num_L2]=[None,num_L2], 输出层 num_L2 = 1 而已
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()
