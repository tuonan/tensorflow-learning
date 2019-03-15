import tensorflow as tf
import  numpy as np

x_data = np.random.rand(100)
y_data = x_data*0.1+0.2

#构造一个线性模型  最后训练的结果k接近0.1,b接近0.2
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 定义最下代价函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 炼数成金的视频课程，没有分清epoch,iteration，batch_size区别
# iteration:batch_size个样本训练一次
# epoch:全部样本训练一次（所有样本的正向、反向传播）
# 本次简单示例没有batch_size概念，则这里的step就是iteration概念
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))
# 结果随着迭代,参数分别接近0.1,0.2.网络训练正确。
# 0 [0.06112067, 0.102395765]
# 20 [0.10823004, 0.19504069]
# 40 [0.104631364, 0.19720921]
# 60 [0.10260624, 0.19842952]
# 80 [0.10146663, 0.19911623]
# 100 [0.10082532, 0.19950268]
# 120 [0.10046444, 0.19972014]
# 140 [0.10026135, 0.19984251]
# 160 [0.10014707, 0.19991139]
# 180 [0.10008276, 0.19995013]
# 200 [0.10004658, 0.19997193]