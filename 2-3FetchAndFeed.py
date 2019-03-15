import  tensorflow as tf

# Fetch  允许取多个op（而不是排序sess.run）
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
add2 = tf.add(input1,add)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([mul,add,add2])
    print(result)

# Feed 同时喂多个值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    # feed的数据以字典的形式传入,但是value是用[]括起来的
    print(sess.run(output,feed_dict={input1:[8.],input2:[2.]}))
