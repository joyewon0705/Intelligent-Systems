# -*- coding: utf-8 -*-

import tensorflow as tf

# 2개의 실행매개변수 placeholder 선언
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 연산 텐서들을 정의
# 아래의 연산텐서들을 실행하기 위해서는
# 실행매개변수 x, y의 값을 반드시 전달해야함
add = tf.add(x,y)               # x+y
subtract = tf.subtract(x,y)     # x-y
multiply = tf.multiply(x, y)    # x*y
div = tf.divide(x,y)            # x/y
mod = tf.mod(x,y)               # x%y

with tf.Session() as sess:
    # 실행 매개변수로 전달된 데이터를 변수로 선언
    # placeholder 변수명과 대소문자까지 일치해야함
    
    feed_dict = {x:10, y:5}
    print("tf.add({0}, {1}) -> {2}".format(10, 5, 
          sess.run(add, feed_dict=feed_dict)))
    
    feed_dict = {x:100, y:50}
    print("tf.subtract({0}, {1}) -> {2}".format(100, 50, 
          sess.run(subtract, feed_dict=feed_dict)))
    
    feed_dict = {x:7, y:3}
    print("tf.multiply({0}, {1}) -> {2}".format(7, 3, 
          sess.run(multiply, feed_dict=feed_dict)))
    
    feed_dict = {x:10, y:3}
    print("tf.div({0}, {1}) -> {2}".format(10, 3, 
          sess.run(div, feed_dict=feed_dict)))    
    
    feed_dict = {x:10, y:5}
    print("tf.mod({0}, {1}) -> {2}".format(10, 5, 
          sess.run(mod, feed_dict=feed_dict)))
    
    
    
    
    
    
    
    
    

