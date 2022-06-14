# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

iris = load_iris()

X_df = pd.DataFrame(iris.data)
y_df = pd.Series(iris.target)

print(X_df.info())
print(X_df.describe())

print(y_df.value_counts() / len(y_df))

X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values,
                     stratify=y_df.values, random_state=1)

X = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.int32, shape=[None])

# 텐서플로우의 다중 클래스 분류는 
# 라벨데이터의 원핫 인코딩을 요구하기 때문에
# 라벨데이터의 전처리가 필요함

# 텐서플로우의 one_hot 함수 사용
# 입력된 텐서를 class의 개수의 맞춰 
# 원핫인코딩된 행렬을 반환
y_one_hot = tf.one_hot(y, 3)

# 텐서플로우의 one_hot 함수는 3차원 텐서를 반환하기 때문에
# 반드시 2차원으로 형태를 변환해야함
print(y_one_hot)
y_one_hot = tf.reshape(y_one_hot, [-1, 3])
print(y_one_hot)

W = tf.Variable(tf.random_normal(shape=[4,3]))
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

















