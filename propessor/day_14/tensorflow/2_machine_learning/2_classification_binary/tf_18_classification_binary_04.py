# -*- coding: utf-8 -*-

import numpy as np

# 이진 분류를 위한 데이터 셋 정의
X_data = [1,2,3,4,5,6,7,8,9,10]
y_data = [0,0,0,0,0,1,1,1,1,1]

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.ones(shape=[1]))
b = tf.Variable(tf.ones(shape=[1]))

# 이진 분류 모델의 가설 정의
h = tf.sigmoid(X * w + b)

# 이진 분류 모델의 손실함수 정의
loss = tf.reduce_mean(tf.square(
        y * -tf.log(h) + (1 - y) * -tf.log(1 - h)))

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# 이진 분류 모델의 예측 값을 반환하기 위한 변수
predicted = tf.cast(h >= 0.5, tf.float32)

# 이진 분류 모델의 정확도의 값을 반환하는 변수
accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted, y), tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_data, y : y_data}

    step = 1
    prev_loss = None
    while True :
        sess.run(train, feed_dict=feed_dict)
        
        loss_value = sess.run(loss, feed_dict=feed_dict)
        
        if step % 100 == 0 :     
            acc_value = sess.run(accuracy, feed_dict=feed_dict)
            print("{0} -> loss : {1:.2f}, acc : {2:.2f}".format(step, loss_value, acc_value))
            
        if prev_loss == None :
            prev_loss = loss_value
        elif prev_loss < loss_value or np.isnan(loss_value) :
            break;
        else:
            prev_loss = loss_value
            
        step += 1    
        
    loss_value, acc_value = sess.run([loss, accuracy], feed_dict=feed_dict)
    print("(결과) -> loss : {1}, acc : {2}".format(step, loss_value, acc_value))
            























