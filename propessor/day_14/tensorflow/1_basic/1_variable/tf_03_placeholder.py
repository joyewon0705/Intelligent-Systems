# -*- coding: utf-8 -*-

# 텐서플로우의 텐서 타입
# 1. 상수(constant)
# 2. 변수(Variable)
# 3. 실행매개변수(placeHolder)

import tensorflow as tf

# 텐서플로우의 실행매개변수 텐서 선언
# tf.placeholder를 사용하여 정의할 수 있음
# 세션을 통해서 run 메소드가 실행될 때, 값이 전달해야하는 텐서
# 각각의 run 메소드 실행 시, 다른 값을 사용하여 실행할 수 있음
# 일반적으로 학습데이터, 테스트 데이터를 전달하기 위해서 사용
# 실행매개변수 텐서를 사용하는 경우 반드시 세션의 run 메소드 매개변수 중,
# feed_dict를 전달해야만 에러가 발생하지 않음
# (placeholder 텐서의 값이 필요하지 않으면 생략해도 됨)
ph_1 = tf.placeholder(tf.float32)

ph_2 = tf.placeholder(tf.float32, 
                      shape=[None])

ph_3 = tf.placeholder(tf.float32, 
                      shape=[None, 2])

# tf.square 함수
# 매개변수로 전달된 텐서의 
# 제곱 값을 반환하는 연산 텐서를 반환
squar_tensor_1 = tf.square(ph_1)

squar_tensor_2 = tf.square(ph_2)

# tf.add 함수
# 매개변수로 전달된 텐서들의 합계를 반환하는 연산 텐서를 반환
plus_tensor = \
    tf.add(squar_tensor_1, squar_tensor_2)

# placeholder 타입의 텐서는 값의 수정이 허용되지 않습니다.
#ph_1_assign = ph_1.assign(11)

# 자동 종료를 위한 세션 객체 생성 시, with절 활용
with tf.Session() as sess :
    # placeholder 타입의 텐서가 활용되는 텐서를 구동하기 위해서
    # feed_dict 매개변수를 전달해야 합니다.
    result = sess.run(
        ph_1, feed_dict={ph_1 : 17})
    print(f"result = {result}")
    
    result = sess.run(
        ph_2, 
        feed_dict={ph_2 : [1,2,3,5,6,7,8,9,10,11,12]})
    print(f"result = {result}")
    
    result = sess.run(ph_3, 
                      feed_dict={
                          ph_3 : 
                              [[1,2],[3,4],[5,6]]})
    print(f"result = {result}")
    
    # result = sess.run(plus_tensor)
    # print(f"result = {result}")
    
    # 실행할 텐서의 연관된 모든 placeholder 타입의 텐서들의
    # 값을 feed_dict로 전달해야만 에러가 발생하지 않습니다.    
    result = sess.run(plus_tensor, 
                      feed_dict={
                          ph_1 : 17,
                          ph_2 : [10,20]})
    print(f"result = {result}")


















