# -*- coding: utf-8 -*-

import tensorflow as tf

# 학습 데이터
X_train = [10., 20, 30, 40, 50]
y_train = [5., 7, 15, 20, 25]

# 테스트 데이터
X_test = [60., 70, 80]
y_test = [32., 38, 40]

# 1. 데이터를 전달받기 위한 실행매개변수 선언
X = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# 2. 학습을 통해 갱신할 변수 텐서의 선언
# - 가중치와 절편의 Variable 텐서 선언
# - 머신러닝 / 딥러닝
# - 입력데이터 X에 대한 가중치/절편을 계산하여
#   y 데이터를 예측하는 것이 최종 목표

# 선형회기를 위한 선형방정식
# - x1 * w1 + x2 * w2 ... xN * wN + b
# - X * W + b = y

# 모든 가중치, 절평에 관련된 변수들은
# 반드시 tf.Variable 을 사용하여 선언합니다.
# 텐서플로우가 실행되는동안 지속적으로
# 값을 갱신(학습)하기 때문에

# 가중치(기울기) 텐서
w = tf.Variable(0.0)

# 절편(편향) 텐서
b = tf.Variable(0.0)

# 3. 머신러닝/딥러닝의 가설 텐서 선언
# - scikit-learn의 예측기 클래스가 제공하는
#   predict 메소드의 값을 생성할 수 있는 식 
# - 문제 해결을 위한 식

# 1차 선형 방정식
# [10, 20, 30, 40, 50] * w + b
# [10 * w, 20 * w, 30 * w, 40 * w, 50 * w] + b
# [10 * w + b, 20 * w + b, 30 * w + b, 40 * w + b, 50 * w + b] 
h = X * w + b

# 4. 손실(오차)의 값을 계산할 수 있는 텐서 선언
# - 선형회귀에서의 오차 계산
# - 평균제곱오차
# - 실제 정답과 예측한 값의 차를 제곱하여 
#   평균을 계산하는 방식

# 텐서플로우의 square 함수를 사용하여
# 실제 정답과 예측 값의 차를 제곱
# 실행 결과로는
# 실제 정답(y) [10, 20, 30], 예측 결과(h) [5, 10, 15]
# square 결과 : [25, 100, 225]
square = tf.square(y - h)

# 텐서플로우의 reduce_mean 함수를 사용하여
# 제곱된 오차의 평균을 계산
# tf.reduce_mean([25, 100, 225]) -> 
# 83.33 오차 값을 반환
reduce_mean = tf.reduce_mean(square)

# 오차 계산을 하나의 텐서로 선언
loss = tf.reduce_mean(tf.square(y - h))

# 5. 경사하강법 객체의 선언과 손실의 값을
# 줄여나가는 방향으로 학습을 할 수 있는 텐서 선언

# - 학습을 위한 경사하강법 구현 객체 텐서
# - tf.train.GradientDescentOptimizer 클래스
# - learning_rate에 지정된 값(기본적으로 0.01)을 
#  사용하여 학습을 진행할 수 있는 클래스
# - tf.Variable로 선언된 텐서의 값을 조정하여
#   minimize 메소드의 매개변수로 전달된 텐서의 값이
#  작아지도록 조정하는 역할
# - X 데이터의 정규화(MinMax, Standard, Robust)
# - GradientDescentOptimizer 객체를 사용하여 
#   적절한 learning_rate을 테스트해야함.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

# learning_rate을 스스로 조정하며 최적의 learning_rate
# 찾아서 학습을 진행할 수 있는 클래스
# - tf.train.AdamOptimizer 클래스
optimizer = tf.train.AdamOptimizer()

# 경사하강법  객체를 사용하여 loss의 값이 줄어들 수 있도록
# 학습을 진행할 텐서를 생성
train = optimizer.minimize(loss)

# tf.Variable로 선언된 텐서의 초기화를 
# 할 수 있는 텐서 선언
init_op = tf.global_variables_initializer()

# 세션 객체를 사용하여 학습을 진행
with tf.Session() as sess :
    sess.run(init_op)
    
    feed_dict = {X:X_train, y:y_train}
    
    for step in range(1, 100001) :
        sess.run(train, feed_dict=feed_dict)
        
        if step % 5 == 0 :
            w_val, b_val, loss_val = sess.run(
                    [w, b, loss], 
                    feed_dict=feed_dict)
            print(step, w_val, b_val, loss_val)

    w_val, b_val, loss_val = sess.run(
                    [w, b, loss], 
                    feed_dict=feed_dict)
    print('학습 종료')
    print(w_val, b_val, loss_val)

    feed_dict = {X:X_test, y:y_test}
    pred, loss_val = sess.run(
                    [h, loss], 
                    feed_dict=feed_dict)
    print('테스트 결과')
    print('실제 정답 : ', y_test)
    print('예측 : ', pred)
    print('오차 : ', loss_val)
            
    from matplotlib import pyplot as plt
    
    plt.scatter(X_train, y_train)
    plt.plot(X_train, sess.run(h, feed_dict=feed_dict))
    plt.show()
    
    feed_dict={X:X_test, y:y_test}
    plt.scatter(X_test, y_test)
    plt.plot(X_test, sess.run(h, feed_dict=feed_dict))
    plt.show()













