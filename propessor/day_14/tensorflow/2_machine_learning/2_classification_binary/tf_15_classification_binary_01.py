# -*- coding: utf-8 -*-

# 이진 분류를 위한 데이터 셋 정의
X_data = [1,2,3,4,5,6,7,8,9,10]
y_data = [0,0,0,1,0,1,1,1,1,1]

import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.ones(shape=[1]))
b = tf.Variable(tf.ones(shape=[1]))

# 텐서플로우를 활용하여
# 이진 데이터를 분류하는 방법
# 1. 선형방정식을 사용하여 값을 예측 (X * w + b)
# 2. 예측한 값을 활성화 함수를 사용하여
#    지정된 영역 내부로 값을 압축
# 3. 활성화 함수별로 지정된 기준값을
#    사용하여 분류값을 예측
#    (sigmoid 함수의 경우 0.5가 기준값)

# 선형방정식을 사용하여 값을 예측
pre_h = X * w + b

# 활성화 함수를 사용하여 지정된 영역 내부로 값을 압축
h = tf.sigmoid(pre_h)

# 양성데이터(1인 경우)의 오차를 계산
# - sigmoid 함수의 실행 결과과 1에 가까질수록
#  오차의 값을 작게 측정하기 위해서
#  -tf.log 의 결과를 사용함
loss_1 = y * -tf.log(h)

# 음성데이터(0인 경우)의 오차를 계산
# - sigmoid 함수의 실행 결과과 0에 가까질수록
#  오차의 값을 작게 측정하기 위해서 
#  1 - h의 값을 이용하여 -tf.log 의 결과를 사용함
loss_0 = (1 - y) * -tf.log(1 - h)

# 음성, 양성 데이터에 대한 오차의 값들을
# 하나의 텐서 변수로 통합
loss_sum = loss_0 + loss_1

# 오차를 제곱한 후, 평균값을 계산
#loss = tf.reduce_mean(tf.square(loss_sum))

# 위의 오차식을 한 줄의 실행문으로 작성하는 예제
loss = tf.reduce_mean(tf.square(
        y * -tf.log(h) + (1 - y) * -tf.log(1 - h)))

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# sigmoid 함수의 실행 결과를 활용하여
# 이진 분류의 값을 반환하는 텐서 변수 선언
predicted = h >= 0.5
predicted_cast = tf.cast(predicted, tf.float32)

accuracy1 = tf.equal(predicted_cast, y)
accuracy2 = tf.cast(accuracy1, tf.float32)
accuracy3 = tf.reduce_mean(accuracy2)

accuracy = tf.reduce_mean(
        tf.cast(tf.equal(predicted_cast, y), tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_data, y : y_data}
    
    for step in range(1, 20001) :
        sess.run(train, feed_dict=feed_dict)
        
        if step % 10 == 0 :
            loss_val, acc_val = sess.run([loss, accuracy],
                                         feed_dict=feed_dict)
            print("step-{0} loss : {1:.2f}, acc : {2:.2f}".format(
                    step, loss_val, acc_val))
    
#    print(sess.run(pre_h, feed_dict=feed_dict))
#    print(sess.run(h, feed_dict=feed_dict))
#    print(sess.run(predicted, feed_dict=feed_dict))
#    print(sess.run(predicted_cast, feed_dict=feed_dict))
    
#    print(sess.run(loss_1, feed_dict=feed_dict))
#    print(sess.run(loss_0, feed_dict=feed_dict))
#    print(sess.run(loss_sum, feed_dict=feed_dict))
#    print(sess.run(loss, feed_dict=feed_dict))
    
#    print(sess.run(accuracy1, feed_dict=feed_dict))
#    print(sess.run(accuracy2, feed_dict=feed_dict))
#    print(sess.run(accuracy3, feed_dict=feed_dict))    
#    print(sess.run(accuracy, feed_dict=feed_dict))













