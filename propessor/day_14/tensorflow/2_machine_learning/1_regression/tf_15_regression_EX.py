# -*- coding: utf-8 -*-

# 사이킷 런의 load_diabetes 데이터를 분석하여
# 결과를 확인하세요
# (텐서플로우의 결과와 사이킷 런의 예측기를 활용한 결과를
#  비교하세요)

# 1. 데이터 로딩 및 확인 과정
import pandas as pd
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X_df = pd.DataFrame(diabetes.data)
y_df = pd.Series(diabetes.target)

pd.options.display.max_columns = 100

print(X_df.info())
print(X_df.describe())

print(y_df.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values,
                     random_state=1)
    
import tensorflow as tf

# 2. 데이터 입력을 위한 실행매개변수(placeholder 변수)
X = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None])

# 3. 가중치(기울기), 절편의 텐서를 선언
W = tf.Variable(tf.zeros(shape=[10, 1]))
b = tf.Variable(0.0)

# 4. 가설(예측식)
# - 오차를 계산하기 위해서 1차원 텐서로 형변환
h = tf.reshape(tf.matmul(X, W) + b, [-1])

# 5. 오차 계산(퍙균제곱오차)
loss = tf.reduce_mean(tf.square(y - h))

# 6. 손실을 감소시키는 방향으로 학습을 진행하기 위한
#    OPTIMIZER 객체 선언
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer = tf.compat.v1.train.AdamOptimizer()

# 7. OPTIMIZER 객체를 사용하여 학습을 진행시키기 위한
#   텐서 선언
train = optimizer.minimize(loss)

# 8. TensorFlow의 세션 객체를 생성하여
#   학습을 진행하고 결과를 확인
with tf.Session() as sess :
    # 8-1. Variable 초기화
    sess.run(tf.global_variables_initializer())
    
    # 8-2. 반복을 통해서 학습을 진행
    feed_dict = {X : X_train, y : y_train}
    step = 1
    prev_loss = None
    while True :
        sess.run(train, feed_dict=feed_dict)
        
        loss_value = sess.run(loss, feed_dict=feed_dict)
        
        if step % 100 == 0 :        
            print("{0} loss : {1}".format(step, loss_value))
            
        if prev_loss == None :
            prev_loss = loss_value
        elif prev_loss < loss_value :
            break;
        else:
            prev_loss = loss_value
            
        step += 1
            
    # 8-3. 학습이 종료된 후, 검증 함수를 사용하여
    # 학습 데이터, 테스트 데이터에 대한 검증을 수행
    from sklearn.metrics import r2_score
   
    # 학습 데이터에 대한 검증
    feed_dict = {X : X_train}
    predicted = sess.run(h, feed_dict=feed_dict)
    print("학습 결과 : ", r2_score(y_train, predicted))
   
    # 테스트 데이터에 대한 검증
    feed_dict = {X : X_test}
    predicted = sess.run(h, feed_dict=feed_dict)
    print("테스트 결과 : ", r2_score(y_test, predicted))

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
print("학습 결과 : ", model.score(X_train, y_train))
print("테스트 결과 : ", model.score(X_test, y_test))






















