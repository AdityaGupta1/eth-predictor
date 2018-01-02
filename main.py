import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[4, 2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4, 1], name="y-input")

theta1 = tf.Variable(tf.random_uniform([2, 2], -1, 1), name="theta1")
theta2 = tf.Variable(tf.random_uniform([2, 1], -1, 1), name="theta2")

bias1 = tf.Variable(tf.zeros([2]), name="bias1")
bias2 = tf.Variable(tf.zeros([1]), name="bias2")

a2 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
hypothesis = tf.sigmoid(tf.matmul(a2, theta2) + bias2)

cost = tf.reduce_mean(((y_ * tf.log(hypothesis)) + ((1 - y_) * tf.log(1.0 - hypothesis))) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0], [1], [1], [0]]

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)
sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 1000 == 0:
        print('epoch ', i)
        print('hypothesis ', sess.run(hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print('theta1 ', sess.run(theta1))
        print('bias1 ', sess.run(bias1))
        print('theta2 ', sess.run(theta2))
        print('bias2 ', sess.run(bias2))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y}))
        print()
