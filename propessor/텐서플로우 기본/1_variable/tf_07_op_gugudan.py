# -*- coding: utf-8 -*-

import tensorflow as tf

# 구구단의 단수 및 곱해지는 수를 증가 시키기 위한 상수
nStep = tf.constant(1)

# 구구단의 시작 단수
dan = tf.Variable(2)

# 구구단의 곱해지는 수
mul = tf.Variable(1)

# 현재 단수에서 1증가된 값을 반환하는 텐서
addDan = tf.add(dan, nStep)
# 현재 곱해지는 수에서 1증가된 값을 반환하는 텐서
addMul = tf.add(mul, nStep)

# 현재 단수를 1 증가시킨 값으로 수정하는 텐서
updateDan = tf.assign(dan, addDan)
# 현재 곱해지는 수를 1 증가시킨 값으로 수정하는 텐서
updateMul = tf.assign(mul, addMul)

# 현재 단수와 곱해지는 연산하여 구구단의 결과를 
# 반환하는 텐서
gugudan_result = tf.multiply(dan, mul)

# 곱해지는 수를 초기화하는 텐서
initMul = tf.assign(mul, nStep)

# 텐서플로우의 모든 Variable 변수들을 초기화하기 위한
# 연산 텐서를 정의
init_variables = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_variables)
    
    for i in range(2, 10):
        
        print("{0} 단을 출력합니다.\n".format(
            sess.run(dan)))
        
        sess.run(initMul)
        for j in range(9):
            gugudan_str = "{0} * {1} = {2}".format(sess.run(dan), 
                   sess.run(mul), sess.run(gugudan_result))
            print(gugudan_str)
            sess.run(updateMul)
        
        sess.run(updateDan)
        print("\n")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    