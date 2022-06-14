# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c='white',
            marker='o', edgecolor='black',
            s=50)
plt.grid()
plt.show()

from sklearn.cluster import KMeans

# 최적의 군집(클러스터)의 개수를 검색하는 방법
# - 엘로우 방법을 활용하여 처리할 수 있음
values = []

for i in range(1, 11) :

    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    
    # 클래스터 내의 각 클래스의 SSE 값을 반환하는
    # inertia_ 속성 값
    values.append(km.inertia_)
    
print(values)

plt.plot(range(1, 11), values, marker='o')
plt.xlabel('numbers of cluster')
plt.ylabel('inertia_')
plt.show()












