# -*- coding: utf-8 -*-

# MNIST 데이터 셋
# 0 ~ 9 까지의 손글씨 이미지 데이터
# 28 * 28 크기의 이미지 데이터(784개의 일차원 데이터)
from tensorflow.examples.tutorials.mnist import input_data

save_dir = '../../../data/MNIST/'
mnist = input_data.read_data_sets(save_dir, one_hot=True)

print(mnist.train.images[1])
print(mnist.train.labels[1])

from matplotlib import pyplot as plt

plt.imshow(mnist.train.images[1].reshape(28,28), 
           cmap='Greys')
plt.show()









