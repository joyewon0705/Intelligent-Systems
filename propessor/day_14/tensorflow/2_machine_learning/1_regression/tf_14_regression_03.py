# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# 텐서플로우를 활용한 선형회기
# X 데이터의 특성이 다수개인 경우

# 학습데이터
# 키와 성별 정보
X_train = np.array([
    [158, 1],
    [170, 1],
    [183, 1],
    [191, 1],
    [155, 0],
    [163, 0],
    [180, 0],
    [158, 0],
    [170, 0]
])
y_train = np.array([64, 86, 84, 80, 
                    49, 59, 67, 54, 67])

# 학습데이터를 전달받기 위한 실행매개변수 선언
# 다차원 배열을 실행매개변수로 전달받는 경우
# 행의 개수에 상관없이 특성이 2개인 데이터를 전달
X = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None])

# 선형회기를 위한 선형방정식
# x1 * w1 + x2 * w2 ... xN * wN + b
# X * W + b = y

# 가중치 변수의 벡터 선언
# (X의 각 특성에 가중치 변수를 부여)
# X 데이터의 형태 : 9 X 2
# W 데이터의 형태 : 2 X 1
# 결과 데이터의 형태 : 9 X 1
W = tf.Variable(tf.ones(shape=[2,1]))

# 절편
b = tf.Variable(0.0)

# 가설(문제 해결을 위한 식)
# 행렬곱을 사용한 예측 값 계산
# 9 X 2 * 2 X 1 -> 9 X 1
# 2차원 형태의 텐서가 반환
h = tf.matmul(X, W) + b

# 손실 계산을 위한 h 텐서의 형 변환
# y 데이터가 1차원이므로 1차원 텐서로 변환
h_reshape = tf.reshape(h, [-1])

# 전치를 활용한 형변환
# h_reshape = tf.transpose(h)

loss = tf.reduce_mean(tf.square(y - h_reshape))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.train.AdamOptimizer()

train = optimizer.minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init_op)
    
    feed_dict={X:X_train, y:y_train}  
    
#    print(sess.run(h, feed_dict=feed_dict))
#    print(sess.run(h_reshape, feed_dict=feed_dict))
#    print(sess.run(y-h, feed_dict=feed_dict))
#    print(sess.run(y-h_reshape, feed_dict=feed_dict))
    
    # 특정 횟수를 반복을 지정하지 않고
    # 오차의 값이 기준으로 하여
    # 학습을 제어하는 코드
    step = 1
    prev_loss = None
    while True :    
        sess.run(train, feed_dict=feed_dict)        
        if step % 10 == 0 :
            w1_val, w2_val, b_val, loss_val = \
            sess.run([W[0], W[1], b, loss], 
                     feed_dict=feed_dict)
            print(step, w1_val, w2_val, b_val, loss_val)
            
        cur_loss = sess.run(loss, 
                            feed_dict=feed_dict)
        if prev_loss == None :
            prev_loss = cur_loss
        elif prev_loss < cur_loss :
            break
        else :
            prev_loss = cur_loss
        
        step += 1
            
    from sklearn.metrics import r2_score
    
    predicted = sess.run(h, feed_dict=feed_dict)
    print("r2 점수 : ", r2_score(y_train, predicted))    
    print("실제 정담 : ", y_train)
    print("예측 : ", predicted)
    
    
    
    
    
    
    
            
    





















