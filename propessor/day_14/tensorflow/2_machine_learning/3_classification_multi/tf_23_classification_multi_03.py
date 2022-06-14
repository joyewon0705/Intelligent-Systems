# -*- coding: utf-8 -*-

# 사이킷 런에서 제공하는 load_wine 데이터를
# 텐서플로우를 사용하여 분석하세요.

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import tensorflow as tf

wine = load_wine()

X_df = pd.DataFrame(wine.data)
y_df = pd.Series(wine.target)

print(X_df.info())  # 13개의 특성으로 구성된 데이터
print(X_df.describe())

print(y_df.value_counts() / len(y_df))

X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values,
                     stratify=y_df.values, random_state=1)
    
# 데이터 전처리(스케일 조정)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#scaler = MinMaxScaler()
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.int32, shape=[None])

# 원핫 인코딩을 위해서 분류할 클래스의 개수를 
# depth 매개변수로 전달
y_one_hot = tf.one_hot(y, depth=3)

W = tf.Variable(tf.random_normal(shape=[13, 3]))
b = tf.Variable(tf.zeros(shape=[3]))

h = tf.nn.softmax(tf.matmul(X, W) + b)

loss = tf.reduce_mean(tf.square(
        tf.reduce_sum(y_one_hot * -tf.log(h), axis=1)))

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

predicted = tf.argmax(h, axis=1)

correct = tf.equal(predicted, tf.argmax(y_one_hot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_train, y : y_train}    

    for step in range(1, 15001) :
        sess.run(train, feed_dict=feed_dict)
        
        loss_val, acc_val = sess.run(
                [loss, accuracy], feed_dict=feed_dict)
        
        if step % 10 == 0 :
            print("{0} : loss -> {1:.2f}, acc -> {2:.2f}".format(step, loss_val, acc_val))
    
    feed_dict = {X : X_test, y : y_test}               
    loss_val, acc_val = sess.run(
                [loss, accuracy], feed_dict=feed_dict)
    print("테스트 : loss -> {0:.2f}, acc -> {1:.2f}".format(loss_val, acc_val))

















