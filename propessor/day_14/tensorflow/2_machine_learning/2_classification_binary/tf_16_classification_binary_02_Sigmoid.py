# -*- coding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot as plt

value = tf.placeholder(tf.float32)
sigmoid_value = tf.sigmoid(value)

sess = tf.Session()

start = -7
step = 0.1
sigmoid_x = []
sigmoid_y = []
while start <= 7 :
    sigmoid_x.append(start)
    sigmoid_y.append(sess.run(sigmoid_value, feed_dict={value:start}))
    start += step
    
plt.title("tf.sigmoid")
plt.xlabel("data")
plt.ylabel("sigmoid value")
plt.plot(sigmoid_x, sigmoid_y, "b--")
plt.hlines(0.5, -7, 7, 
           colors='y', linestyles='dashed')
plt.show()    
