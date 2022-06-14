# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_df = pd.DataFrame(cancer.data)
y_df = pd.Series(cancer.target)

pd.options.display.max_columns = 100

print(X_df.info())
print(X_df.describe())

print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values,
                     stratify=y_df.values, random_state=1)
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None])

# 가중치 행렬 텐서
W = tf.Variable(tf.random_normal([30, 1]))

# 절편 텐서
b = tf.Variable(0.0)

# 가설(예측) 식의 정의
# h의 shape는????? -> 2차원 텐서
# None X 30 * 30 X 1 -> None X 1
h = tf.sigmoid(tf.matmul(X, W) + b)

# 행렬 곱의 결과는 2차원 텐서가 반환되므로
# 1차원의 정답 데이터(y)와 연산이 올바르게 처리되지 않습니다.
# 2차원 텐서의 shape를 1차원 텐서로 변경하여
# 오차의 값을 정확히 계산할 수 있도록 함
h_reshape = tf.reshape(h, [-1])

# 손실함수의 정의
loss = tf.reduce_mean(tf.square(        
        y * -tf.log(h_reshape) + (1-y) * -tf.log(1-h_reshape)))

# 학습 객체 선언
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# 예측 값 반환 텐서
predicted = tf.cast(h_reshape >= 0.5, tf.float32)

# 정확도 반환 텐서
accuracy = tf.reduce_mean(tf.cast(
        tf.equal(predicted, y), tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_train, y : y_train}
    
    step = 1
    prev_loss = None    
    prev_w = None
    prev_b = None
    while True :
        sess.run(train, feed_dict=feed_dict)
        
        loss_val, acc_val, w_val, b_val = \
            sess.run([loss, accuracy, W, b], feed_dict=feed_dict)
            
        if step % 100 == 0 :
            print("{0} step : loss -> {1:.2f}, acc -> {2:.2f}".format(
                    step, loss_val, acc_val))
            
        if prev_loss == None or prev_loss > loss_val :
            prev_loss = loss_val
            prev_w = w_val
            prev_b = b_val            
        elif loss_val > prev_loss or np.isnan(loss_val) :
            # 직전의 가중치, 절편의 값으로 복원
            sess.run(tf.assign(W, prev_w))
            sess.run(tf.assign(b, prev_b))
            break
        
        step += 1
        
    feed_dict = {X : X_train, y : y_train}
    loss_val, acc_val = \
            sess.run([loss, accuracy], feed_dict=feed_dict)
    print("학습 결과 : ", loss_val, ", ", acc_val)
    
    feed_dict = {X : X_test, y : y_test}
    loss_val, acc_val = \
            sess.run([loss, accuracy], feed_dict=feed_dict)
    print("테스트 결과 : ", loss_val, ", ", acc_val)




















