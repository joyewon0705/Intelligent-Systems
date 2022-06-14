# -*- coding: utf-8 -*-

# 텐서플로우의 변수 텐서의 타입
# 1. 상수(constant)
# 2. 변수(Variable)
# 3. 실행매개변수(placeHolder)

import tensorflow as tf

# 텐서플로우의 상수 텐서 선언
# tf.constant를 사용하여 정의할 수 있음
# 프로그램의 진행 중 값의 변경이 허용되지 않는 변수
# 상수 텐서는 반드시 초기화가 진행되야만 에러가 발생하지 않음
cons_1 = tf.constant(1)

# - 텐서플로우의 모든 변수는 텐서
# - 텐서 : 변수, 연산자, 특정 텐서를 참조하는 텐서
# - 텐서의 모습(값, 실행의 결과)을 확인하는 방법
#   텐서플로우의 세션을 통해서만 가능
print(cons_1)



cons_2 = tf.constant([2.0])
print(cons_2)

cons_3 = tf.constant([1,2,3])

cons_4 = tf.constant([1,2,3,4], shape=[2,2], dtype=tf.float64)

# 상수 텐서는 값의 수정이 허용되지 않습니다.
# 텐서플로우의 수정 메소드
# tf.assign(수정할텐서, 수정할 값)
#cons_1_assign = tf.assign(cons_1, 10)  

sess = tf.Session()

result = sess.run(cons_1)
print(f"result = {result}")

result = sess.run(cons_2)
print(f"result = {result}")

result = sess.run(cons_3)
print(f"result = {result}")

result = sess.run(cons_4)
print(f"result = {result}")
print(f"result = {result[0]}")
print(f"result = {result[1]}")

sess.close()













