# -*- coding: utf-8 -*-

import tensorflow as tf

# 텐서플로우의 텐서 형태 변경
# tf.reshape을 사용하여 특정 텐서의 형태(차원)을 변경할 수 있음
# -1 값을 활용할 수 있음

# tf.zeros(형태) -> 모든 요소가 0으로 채워진 배열이 반환
# [0,0,0,0,0,0,0,0,0,0]
X = tf.Variable(tf.zeros([10]))

# tf.reshape(텐서, [변경할형태])
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
    
    result = sess.run(X)
    print(f"result = {result}")
    
    result = sess.run(X_reshape)
    print(f"result = {result}")










