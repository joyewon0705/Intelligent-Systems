# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

fname = '../../../data/diabetes.csv'
diabetes = pd.read_csv(fname, header=None)

X_df = diabetes.iloc[:, :-1]

print(X_df.info())
pd.options.display.max_columns = 100
print(X_df.describe())

y_df = diabetes.iloc[:, -1]

print(y_df.head())
print(y_df.value_counts() / len(y_df))

X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     stratify=y_df.values, random_state=1)
    
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

print(X_train.shape)
print(X_test.shape)
   
import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None, 164])
y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([164, 1]))
b = tf.Variable(0.0)

h = tf.reshape(tf.sigmoid(tf.matmul(X, W) + b), [-1])

loss = tf.reduce_mean(tf.square(
        y * -tf.log(h) + (1 - y) * -tf.log(1 - h)))

train = tf.train.AdamOptimizer().minimize(loss)

predicted = tf.cast(h >= 0.5, tf.float32)

accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted, y), tf.float32))

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
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 학습 데이터 검증
    feed_dict = {X : X_train}
    pred = sess.run(predicted, feed_dict=feed_dict)
    print("학습 데이터의 confusion_matrix")
    print(confusion_matrix(y_train, pred))
    print("학습 데이터의 classification_report")
    print(classification_report(y_train, pred))
    
    # 테스트 데이터 검증
    feed_dict = {X : X_test}
    pred = sess.run(predicted, feed_dict=feed_dict)
    print("테스트 데이터의 confusion_matrix")
    print(confusion_matrix(y_test, pred))
    print("테스트 데이터의 classification_report")
    print(classification_report(y_test, pred))
    
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     stratify=y_df.values, random_state=1)
    
model = SVC(C=1, gamma=1).fit(X_train, y_train)
print("학습 데이터 평가 : ", model.score(X_train, y_train))
print("테스트 데이터 평가 : ", model.score(X_test, y_test))



















