# -*- coding: utf-8 -*-

# 텐서플로우를 사용하여 다중 클래스를 분류하는 예제

import numpy as np
import tensorflow as tf

# 입력데이터 X
X_train = [ [1,2,1,1],
            [2,1,3,2],
            [5,3,1,7],
            [3,2,1,1],
            [4,2,1,5],
            [1,2,5,5],
            [6,2,6,6]]

# 라벨데이터 y (원핫인코딩으로 처리된 데이터)
# 원핫인코딩 : 대다수의 데이터 0으로 구성되며
# 특정 인덱스의 위치값이 1인 데이터를 의미합니다.
# 원핫인코딩의 특성 개수는 클래스의 수와 일치하며
# 1인 자리의 인덱스가 클래스의 값이 됩니다.
y_train = [ [0,0,1],
            [0,0,1],
            [1,0,0],
            [0,1,0],
            [0,1,0],
            [0,0,1],
            [1,0,0]]

# 텐서플로우 세션의 run 메소드 실행 시
# 데이터를 전달받을 텐서의 선언
X = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

# 가중치 텐서의 선언
# - 다중 클래스 분류의 경우 가중치 텐서의 반환값은
#  클래스의 개수가 됩니다.
W = tf.Variable(tf.random_normal(shape=[4,3]))

# 절편 텐서의 선언
# - 다중 클래스 분류의 경우 각 클래스의 값에 대한
#  절편 값을 할당하기 위해서 클래스의 개수만큼 절편을 생성합니다.
b = tf.Variable(tf.zeros(shape=[3]))

# 가설(예측) 정의

# 텐서플로우의 다항 분류 방법은 
# 각 클래스별 확률 값을 계산하는 방식
# 이항 분류와 다르게 각 클래스 별 확률을 계산하기
# 위해서 출력의 개수는 분류(클래스)의 개수와 일치합니다.
h_matmul = tf.matmul(X, W) + b

# tf.nn.softmax 함수는
# 핼렬 텐서를 입력받아 각 열의 확률값을 
# 반환하는 기능을 제공
# 반환되는 값은 0 ~ 1사이의 값이 반환되며
# 각 행의 총 합계는 1이 됩니다.
# [예시]
# [값1, 값2, 값3] -> [0.5, 0.2, 0.3]
h = tf.nn.softmax(h_matmul)

# 손실 함수의 정의

# 1. 각 예측값을 로그 함수를 사용하여 오차를 계산한 후
# 원핫인코딩과 곱하여 정답 데이터에 대해서만
# 오차 값을 추출
loss_1 = y * -tf.log(h)

# 2. 각 행에 대해서 합계를 계산
# 결과값은 1차원 배열이 됨
# 요소의 개수는 행의 수가 동일함
# axis 매개변수를 사용하지 않는 경우 전체 합계
# axis = 0인 경우 각 열의 합계(열의 개수만큼 반환)
# axis = 1인 경우 각 행의 합계(행의 개수만큼 반환)
loss_2 = tf.reduce_sum(loss_1, axis = 1)

# 3. 각 행의 오차를 합산한 후, 평균을 계산
# - 멀티 클래스의 분류에서의 오차 값
loss_3 = tf.reduce_mean(tf.square(loss_2))

# 오차 함수 정리
loss = tf.reduce_mean(tf.square(
        tf.reduce_sum(y * -tf.log(h), axis=1)))

# 학습 객체 선언
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

# 텐서플로우의 다항 분류 모델의 예측 값을 
# 반환하는 텐서 선언

# tf.argmax 함수는 행렬에서 가장 큰 값의
# 인덱스의 값을 추출하여 반환하는 함수
# axis = 0 인 경우 각각의 열에서 가장 큰 값의
# 행의 인덱스를 추출, axis = 1 인 경우 각각의 행에서
# 가장 큰 값의 열의 인덱스를 반환
predicted = tf.argmax(h, axis=1)

# 정확도 값을 반환하는 텐서 선언

# tf.equal 함수는 두 개의 턴서를 입력받아
# 각 위치를 비교하여 동일한 경우 True, 서로
# 다른 경우 False를 반환(텐서를 반환)
correct = tf.equal(predicted, tf.argmax(y, axis=1))

# tf.cast 함수는 입력된 텐서의 데이터를
# 다른 형으로 변환하는 기능을 제공
# 아래와 같은 경우 Boolean 타입의 값을 실수로 변환
# True -> 1, False -> 0
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess : 
    sess.run(tf.global_variables_initializer())
    
    feed_dict = {X : X_train, y : y_train}
    
#    print(sess.run(h_matmul, feed_dict=feed_dict))
#    print(sess.run(h, feed_dict=feed_dict))
#    print(sess.run(predicted, feed_dict=feed_dict))
#    
#    print(sess.run(loss_1, feed_dict=feed_dict))
#    print(sess.run(loss_2, feed_dict=feed_dict))
#    print(sess.run(loss_3, feed_dict=feed_dict))
    
    for step in range(1, 10001) :
        sess.run(train, feed_dict=feed_dict)
        
        loss_val, acc_val = sess.run(
                [loss, accuracy], feed_dict=feed_dict)
        
        if step % 10 == 0 :
            print("{0} : loss -> {1:.2f}, acc -> {2:.2f}".format(step, loss_val, acc_val))


















