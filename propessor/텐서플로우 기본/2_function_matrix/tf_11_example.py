# -*- coding: utf-8 -*-

import tensorflow as tf

# 텐서플로우를 활용한 머신러닝 예제

# 학습데이터 정의 (회귀분석용 데이터)
X_train = [10, 20, 30, 40, 50]
y_train = [5, 7, 15, 20, 25]

# 1. 데이터를 전달받기 위한 실행매개변수 선언
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# 2. 머신러닝을 위한 가설 정의
# - 선형회기 분석을 위한 선형방정식
# - X * W + b

# 기울기(가중치) 변수 선언
w = tf.Variable(0, dtype=tf.float32)

# 절편(편향) 변수 선언
b = tf.Variable(0, dtype=tf.float32)

# 가설(문제를 해결하기 위한 식)
h = X * w + b

# 머신러닝 모델의 학습을 진행하기 위한
# 오차 값의 계산(평균제곱오차)
loss_1 = y - h
loss_2 = tf.square(loss_1)
loss = tf.reduce_mean(loss_2)

# 오차 값을 감소시키는 방향으로
# 학습을 진행할 수 있는 객체의 선언
# (학습률(learning_rate)을 지정하여 학습의 속도를 제어)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_train, y : y_train}
    for step in range(1, 101):        
        sess.run(train, feed_dict=feed_dict)
        
        if step % 10 == 0 :
            pred_loss = sess.run(loss, feed_dict=feed_dict)
            w_val, b_val = sess.run([w, b], feed_dict=feed_dict)
            print("(step-{0}) 오차 : {1}, w : {2}, b : {3}".format(
                    step, pred_loss, w_val, b_val))
            
    pred_loss = sess.run(loss, feed_dict=feed_dict)
    print("최종 오차 : {}".format(pred_loss))
    pred = sess.run(h, feed_dict=feed_dict)
    print("예측 결과 : ", pred)
    
    from matplotlib import pyplot as plt
    
    plt.plot(X_train, y_train, 'or')
    plt.plot(X_train, pred, '--b')
    
    X_test = [37, 22]
    pred_test = sess.run(h, feed_dict={X : X_test})
    plt.plot(X_test, pred_test, 'xg')
    
    plt.show()
    
    """
    feed_dict = {X : X_train, y : y_train}
    pred = sess.run(h, feed_dict=feed_dict)
    print("예측 결과 : ", pred)
    
    pred_loss = sess.run(loss_1, feed_dict=feed_dict)
    print("오차 값 1 : ", pred_loss)
    
    pred_loss = sess.run(loss_2, feed_dict=feed_dict)
    print("오차 값 2 : ", pred_loss)
    
    pred_loss = sess.run(loss, feed_dict=feed_dict)
    print("오차 값 : ", pred_loss)
    
    # 학습을 진행시키는 코드
    # (오차를 줄여나가는 방향으로 가중치(기울기)와 
    # 졀편의 값을 보정시킴)
    sess.run(train, feed_dict=feed_dict)
    
    pred = sess.run(h, feed_dict=feed_dict)
    print("예측 결과 : ", pred)
    
    pred_loss = sess.run(loss_1, feed_dict=feed_dict)
    print("오차 값 1 : ", pred_loss)
    
    pred_loss = sess.run(loss_2, feed_dict=feed_dict)
    print("오차 값 2 : ", pred_loss)
    
    pred_loss = sess.run(loss, feed_dict=feed_dict)
    print("오차 값 : ", pred_loss)
    """



















