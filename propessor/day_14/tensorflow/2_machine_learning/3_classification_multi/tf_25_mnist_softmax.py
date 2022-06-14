# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

save_dir = '../../../data/MNIST/'
mnist = input_data.read_data_sets(save_dir, one_hot=True)

nFeatures = 784
nClasses = 10

# 784개의 특성으로 구성된 2차원 데이터를 입력받는 텐서
X = tf.placeholder(tf.float32, shape=[None, nFeatures])
# 10개의 클래스를 분류하기 위해서 원핫인코딩된
# 라벨데이터를 입력받는 텐서
y = tf.placeholder(tf.float32, shape=[None, nClasses])

W = tf.Variable(tf.random_normal(shape=[nFeatures, nClasses]))
b = tf.Variable(tf.zeros(shape=[nClasses]))

h = tf.nn.softmax(tf.matmul(X, W) + b)

loss = tf.reduce_mean(tf.square(
        tf.reduce_sum(y * -tf.log(h), axis=1)))

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

predicted = tf.argmax(h, axis=1)

correct = tf.equal(predicted, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X:mnist.train.images, y:mnist.train.labels}

    for step in range(1, 10001) :
        sess.run(train, feed_dict=feed_dict)        
                
        if step % 100 == 0 :
            loss_val, acc_val = sess.run(
                [loss, accuracy], feed_dict=feed_dict)            
            print("{0}-학습데이터 : loss -> {1:.2f}, acc -> {2:.2f}".format(step, loss_val, acc_val))
            
            loss_val, acc_val = sess.run(
                [loss, accuracy], feed_dict={X:mnist.validation.images, y:mnist.validation.labels})            
            print("{0}-검증데이터 : loss -> {1:.2f}, acc -> {2:.2f}".format(step, loss_val, acc_val))
    
    feed_dict = {X:mnist.test.images, y:mnist.test.labels}               
    loss_val, acc_val = sess.run(
                [loss, accuracy], feed_dict=feed_dict)
    print("테스트 : loss -> {0:.2f}, acc -> {1:.2f}".format(loss_val, acc_val))

















