# -*- coding: utf-8 -*-

import tensorflow as tf

X1 = tf.Variable([[1,2],[3,4]]) # 2 X 2
X2 = tf.Variable([[5,6,7,8]])   # 1 X 4

X2_reshape_1 = tf.reshape(X2, [2, -1])  # 2 X 2

# 텐서플로우의 행렬 곱
# tf.matmul을 사용하여 처리할 수 있음
# 2개의 행렬을 저장하고 있는 텐서를 입력받아
# 행렬곱의 결과를 반환하는 함수
matmul_1 = tf.matmul(X1, X2_reshape_1)

X2_reshape_2 = tf.reshape(X2, [4, -1])  # 4 X 1

# 행렬 곱의 규칙에 위배되기 때문에 에러가 발생됨
#matmul_2 = tf.matmul(X1, X2_reshape_2)

init_variables = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init_variables)
    
    result = sess.run(matmul_1)
    print(f"result = {result}")
    
#    result = sess.run(matmul_2)
#    print(f"result = {result}")
    







