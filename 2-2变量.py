import  tensorflow as tf

x = tf.Variable([1,2])
a = tf.constant([3,3])
sub = tf.subtract(x,a)
add = tf.add(x,sub)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

state = tf.Variable(0)
new_value = tf.add(state,1)
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(state))  教程里面是有这行的，去掉不去掉不影响下面的sess.run(update）值
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))