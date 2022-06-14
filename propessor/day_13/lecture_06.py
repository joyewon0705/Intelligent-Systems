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

# KMeans 클래스를 사용
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
km.fit(X)

y_cluster = km.predict(X)

plt.scatter(X[y_cluster == 0, 0],
            X[y_cluster == 0, 1],
            s=50, c='lightgreen',
            marker='s', label='Cluster 1')

plt.scatter(X[y_cluster == 1, 0],
            X[y_cluster == 1, 1],
            s=50, c='orange',
            marker='o', label='Cluster 2')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=100, c='red',
            marker='*', label='Center')

plt.legend()
plt.grid()
plt.show()


# AgglomerativeClustering 사용하여 군집
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2)
y_cluster = ac.fit_predict(X)

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











