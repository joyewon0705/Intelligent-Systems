# -*- coding: utf-8 -*-

import numpy as np

arr1 = np.array([1,2,3,4,5])
temp1 = 10
# 분배 법칙이 성립되어 arr1의 각 요소에
# temp 변수의 값이 더해짐
print(arr1 + temp1)

arr2 = np.array([1,2,3,4,5])
arr3 = np.array([1,2,3,4,5])
# 동일한 차원, 동일한 요소 수를 연산하는 경우
# 각 위치에 해당되는 요소 사이에서 연산이 실행
# 분배 X
print(arr2 + arr3)

arr4 = np.array([1,2,3,4,5])
arr5 = np.array([5])
# 동일한 차원이지만 요소의 개수가 서로 다른 경우
# 개수가 적은 쪽의 모든 요소를 개수가 많은 쪽의 요소에
# 분배하여 연산을 실행합니다.
# 주의사항 - 분배가 성립할 수 있는 개수에서만 실행될 수 있음
print(arr4 + arr5)

import tensorflow as tf

# 텐서플로우 연산의 분배법칙
# 연산의 피연산자들 형태가 동일한 경우 서로 동일한 위치의 요소가 연산됨.
# 반면 서로 형태가 다른 경우 한 텐서의 모든 요소에 연산이 실행됨
# (연산이 불가능한 형태인 경우 에러가 발생)

X = tf.Variable(tf.zeros([10]))
"""
[
 [0,0],
 [0,0],
 [0,0],
 [0,0],
 [0,0]
]
"""
X_reshape = tf.reshape(X, [-1, 2])

init_variables = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init_variables)
    
    # [0,0,0,0,0,0,0,0,0,0]
    result = sess.run(X)
   
    print(f"result = {result}")
    
    # [1,2,3,4,5,6,7,8,9,10]
    result = sess.run(X + list(range(1,11)))    
    print(f"result = {result}")
    
    # X_reshape 텐서의 모든 요소에 대해서 
    # 7을 더한 결과를 반환
    """
    [
     [7,7],
     [7,7],
     [7,7],
     [7,7],
     [7,7]
    ]
    """
    result = sess.run(X_reshape + 7)
    print(f"result = {result}")
    
    # X_reshape 각 행에 대해서 2와 5를 더한
    # 결과를 반환
    """
    [
     [2,5],
     [2,5],
     [2,5],
     [2,5],
     [2,5]
    ]
    """
    result = sess.run(X_reshape + [2,5])
    print(f"result = {result}")
    
    # 5 X 2형태의 텐서에 3의 크기를 갖는 리스트를
    # 더할 수 없습니다.
#    result = sess.run(X_reshape + [2,5,7])
#    print(f"result = {result}")
























