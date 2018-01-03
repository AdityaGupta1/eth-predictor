import tensorflow as tf
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt


def string_to_list(list_to_convert):
    return '[' + ', '.join(str(e) for e in list_to_convert) + ']'


url = urllib.request.urlopen("https://api.gdax.com/products/ETH-USD/candles")
data = url.read()
encoding = url.info().get_content_charset('utf-8')
parsed_data = json.loads(data.decode(encoding))
print('parsedData: ' + string_to_list(parsed_data))

past_data = 50

changes = []

for i in range(len(parsed_data)):
    candle = parsed_data[i]
    prices = [candle[3], candle[4]]
    change = prices[1] - prices[0]
    changes.append(round(change, 3))

print('changes: ' + string_to_list(changes))

eth_x = []
eth_y = []

for i in range(past_data, len(changes)):
    eth_x.append(changes[i - past_data : i])
    eth_y.append([changes[i]])

print('eth_x: ' + string_to_list(eth_x))
print('eth_y: ' + string_to_list(eth_y))

# exit(0)

np.set_printoptions(suppress=True)

x_ = tf.placeholder(tf.float32, shape=[len(eth_x), past_data], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[len(eth_y), 1], name="y-input")

theta1 = tf.Variable(tf.random_uniform([past_data, past_data], -5, 5), name="theta1")
theta2 = tf.Variable(tf.random_uniform([past_data, past_data], -5, 5), name="theta2")
theta3 = tf.Variable(tf.random_uniform([past_data, 1], -5, 5), name="theta3")

bias1 = tf.Variable(tf.zeros([past_data]), name="bias1")
bias2 = tf.Variable(tf.zeros([past_data]), name="bias2")
bias3 = tf.Variable(tf.zeros([1]), name="bias3")

a2 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
a3 = tf.sigmoid(tf.matmul(x_, theta2) + bias2)
hypothesis = tf.matmul(a3, theta3) + bias3

cost = tf.reduce_mean(tf.squared_difference(hypothesis, y_))

train_step = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print()

completed_hypothesis = None;

for i in range(10000):
    sess.run(train_step, feed_dict={x_: eth_x, y_: eth_y})
    if i % 1000 == 0:
        print('epoch ', i)
        completed_hypothesis = sess.run(hypothesis, feed_dict={x_: eth_x, y_: eth_y});
        print('hypothesis ', completed_hypothesis)
        # print('theta1 ', sess.run(theta1))
        # print('bias1 ', sess.run(bias1))
        # print('theta2 ', sess.run(theta2))
        # print('bias2 ', sess.run(bias2))
        print('cost ', sess.run(cost, feed_dict={x_: eth_x, y_: eth_y}))
        print()

flattened_eth_y = [item for items in eth_y for item in items]
flattened_eth_y = [('%.2f' % item).rjust(5) for item in flattened_eth_y]
flattened_hypothesis = [item for items in completed_hypothesis for item in items]
flattened_hypothesis = [('%.2f' % item).rjust(5) for item in flattened_hypothesis]

print(flattened_eth_y)
print(flattened_hypothesis)
