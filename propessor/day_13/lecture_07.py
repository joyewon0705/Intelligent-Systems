# -*- coding: utf-8 -*-

from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, 
                  noise=0.05,
                  random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1])
plt.show()

# DBSCAN 클래스를 사용
# - 데이터간의 밀도를 이용하여 군집의 형성 여부를
#   결정하는 알고리즘을 구현
# - DBSCAN는 군집의 크기가 자동을 결정
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')

y_cluster = db.fit_predict(X)

plt.scatter(X[y_cluster == 0, 0],
            X[y_cluster == 0, 1],
            s=50, c='lightgreen',
            marker='s', label='Cluster 1')

plt.scatter(X[y_cluster == 1, 0],
            X[y_cluster == 1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.legend()
plt.grid()
plt.show()












