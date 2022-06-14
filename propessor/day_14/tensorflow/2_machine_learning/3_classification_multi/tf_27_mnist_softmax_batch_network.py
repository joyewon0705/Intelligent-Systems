# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data

save_dir = '../../../data/MNIST/'
mnist = input_data.read_data_sets(save_dir, one_hot=True)

import tensorflow as tf

nFeatures = 784
nClasses = 10

nFeatures_hidden1 = 392
nFeatures_hidden2 = 196

X = tf.placeholder(tf.float32, shape=[None, nFeatures])
y = tf.placeholder(tf.float32, shape=[None, nClasses])

W1 = tf.Variable(tf.random_normal([nFeatures, nFeatures_hidden1]))
b1 = tf.Variable(tf.random_normal([nFeatures_hidden1]))
hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([nFeatures_hidden1, nFeatures_hidden2]))
b2 = tf.Variable(tf.random_normal([nFeatures_hidden2]))
hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2) + b2)

W = tf.Variable(tf.random_normal([nFeatures_hidden2, nClasses]))
b = tf.Variable(tf.random_normal([nClasses]))

h = tf.nn.softmax(tf.matmul(hidden2, W) + b)

loss = tf.reduce_mean(tf.square(tf.reduce_sum(y * -tf.log(h), axis=1)))

train = tf.train.AdamOptimizer().minimize(loss)

predicted = tf.argmax(h, axis=1)
correct = tf.equal(predicted, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 배치 학습
# - 전체 학습데이터를 사용하여 학습을 진행할 때
#  지역 최소값을 회피할 수 없는 문제점이 발생할 수 있기때문에
#  학습 데이터를 분할하여 각 부분별로 학습을 진행하는 방식
# - 배치 학습을 통해서 학습하는 경우 지역 최소값을 효과적으로
#  회피할 수 있으며, 대량의 데이터 셋을 머신의 성능에 맞춰
#  효율적으로 학습을 시킬 수 있는 장점이 있습니다.

# - epoches : 전체 학습 데이터 셋의 학습 횟수
epoches = 100
# - batch_size : 한번의 학습에서 처리할 샘플의 개수
batch_size = 100
# - iter_num : 전체 학습 데이터를 배치 사이즈로 나누었을때
#              반복을 수행해야하는 횟수를 저장하는 변수
iter_num = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    for epoch_step in range(1, epoches+1) :
        
        loss_avg = 0
        for batch_step in range(1, iter_num + 1) :
            batch_X, batch_y = mnist.train.next_batch(batch_size)
            
            # batch_size에 지정된 샘플의 개수를 사용하여
            # 학습을 진행(확률적 경사하강법과 유사한 방식)
            _, loss_val = sess.run([train, loss], 
                     feed_dict={X:batch_X, y:batch_y})
            
            loss_avg = loss_avg + loss_val / iter_num
            
        print(f"epoch_{epoch_step} : {loss_avg}")   
        
        loss_val, acc_val = sess.run([loss, accuracy], 
                     feed_dict={X:mnist.validation.images,
                                y:mnist.validation.labels})   
        print(f"validation_{epoch_step} : {loss_val}, {acc_val}\n")   
        
        
    loss_val, acc_val = sess.run([loss, accuracy], 
                     feed_dict={X:mnist.test.images,
                                y:mnist.test.labels})

    print(f"TEST : {loss_val}, {acc_val}")











