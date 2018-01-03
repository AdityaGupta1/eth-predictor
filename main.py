import tensorflow as tf
import urllib.request
import json


def string_to_list(list_to_convert):
    return '[' + ', '.join(str(e) for e in list_to_convert) + ']'


url = urllib.request.urlopen("https://api.gdax.com/products/ETH-USD/candles")
data = url.read()
encoding = url.info().get_content_charset('utf-8')
parsed_data = json.loads(data.decode(encoding))
print('parsedData: ' + string_to_list(parsed_data))

past_data = 10

changes = []

for i in range(len(parsed_data)):
    candle = parsed_data[i]
    prices = [candle[1], candle[2]]
    change = prices[1] - prices[0]
    changes.append(change)

print('changes: ' + string_to_list(changes))

eth_x = []
eth_y = []

for i in range(past_data, len(changes)):
    eth_x.append(changes[i - 10 : i])
    eth_y.append(changes[i])

print('eth_x: ' + string_to_list(eth_x))
print('eth_y: ' + string_to_list(eth_y))

exit(0)

x_ = tf.placeholder(tf.float32, shape=[len(eth_x), past_data], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[len(eth_y), 1], name="y-input")

theta1 = tf.Variable(tf.random_uniform([past_data, 10], -1, 1), name="theta1")
theta2 = tf.Variable(tf.random_uniform([10, 1], -1, 1), name="theta2")

bias1 = tf.Variable(tf.zeros([past_data]), name="bias1")
bias2 = tf.Variable(tf.zeros([1]), name="bias2")

a2 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
hypothesis = tf.sigmoid(tf.matmul(a2, theta2) + bias2)

cost = tf.reduce_mean(((y_ * tf.log(hypothesis)) + ((1 - y_) * tf.log(1.0 - hypothesis))) * -1)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    sess.run(train_step, feed_dict={x_: eth_x, y_: eth_y})
    if i % 1000 == 0:
        print('epoch ', i)
        print('hypothesis ', sess.run(hypothesis, feed_dict={x_: eth_x, y_: eth_y}))
        print('theta1 ', sess.run(theta1))
        print('bias1 ', sess.run(bias1))
        print('theta2 ', sess.run(theta2))
        print('bias2 ', sess.run(bias2))
        print('cost ', sess.run(cost, feed_dict={x_: eth_x, y_: eth_y}))
        print()
