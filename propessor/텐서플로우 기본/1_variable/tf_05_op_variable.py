# -*- coding: utf-8 -*-

import tensorflow as tf

x = tf.Variable(10)
y = tf.Variable(7)

add = tf.add(x, y)
subtract = tf.subtract(x, y)
multiply = tf.multiply(x, y)
div = tf.divide(x, y)
mod = tf.mod(x, y)

with tf.Session() as sess:
    # tf.Variable 을 통해 생성된 텐서의 초기화를 
    # 위해서 작성된 코드
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    
    print("add({0}, {1}) -> {2}".format(\
          sess.run(x), sess.run(y), sess.run(add)))
    print("subtract({0}, {1}) -> {2}".format(\
          sess.run(x), sess.run(y), sess.run(subtract)))
    print("multiply({0}, {1}) -> {2}".format(\
          sess.run(x), sess.run(y), sess.run(multiply)))
    print("div({0}, {1}) -> {2}".format(\
          sess.run(x), sess.run(y), sess.run(div)))
    print("mod({0}, {1}) -> {2}".format(\
          sess.run(x), sess.run(y), sess.run(mod)))
    
    
    
    
    