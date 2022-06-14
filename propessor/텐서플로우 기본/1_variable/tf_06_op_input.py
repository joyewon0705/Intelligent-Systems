# -*- coding: utf-8 -*-

import tensorflow as tf

num1 = tf.placeholder(tf.float32)
num2 = tf.placeholder(tf.float32)

add = tf.add(num1, num2)
subtract = tf.subtract(num1, num2)
multiply = tf.multiply(num1, num2)
div = tf.divide(num1, num2)
mod = tf.mod(num1, num2)

with tf.Session() as sess:
    n1 = float(input("첫 번째 숫자를 입력하세요 : "))
    n2 = float(input("두 번째 숫자를 입력하세요 : "))
    
    feed_dict = {num1: n1, num2: n2}

    print("add -> {}".format(\
          sess.run(add, feed_dict=feed_dict)))
    print("subtract -> {}".format(\
          sess.run(subtract, feed_dict=feed_dict)))
    print("multiply -> {}".format(\
          sess.run(multiply, feed_dict=feed_dict)))
    print("div -> {}".format(\
          sess.run(div, feed_dict=feed_dict)))
    print("mod -> {}".format(\
          sess.run(mod, feed_dict=feed_dict)))
    
    
    
    
    
    
    
    
    