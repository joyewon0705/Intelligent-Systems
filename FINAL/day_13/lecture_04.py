# -*- coding: utf-8 -*-

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(X.head())
print(X.info())
print(X.describe())

# 머신러닝 모델의 학습 결과
# 가중치가 할당되지 않은 (또는 가중치가 굉장히 작은 값)
# 컬럼들에 대해서 새로운 특성을 생성
# - 차원축소
# - 군집분석
X_part = X[['radius error', 'compactness error', 'concavity error']]

# 최적의 군집 개수를 검색
# - 5개의 군집 개수가 최적화
# from sklearn.cluster import KMeans
# values = []
# for i in range(1, 15) :
#     km = KMeans(n_clusters=i,
#                 init='k-means++',
#                 n_init=10,
#                 max_iter=300,
#                 random_state=0)
#     km.fit(X_part)
#     values.append(km.inertia_)
    
# print(values)

# import matplotlib.pyplot as plt
# plt.plot(range(1, 15), values, marker='o')
# plt.xlabel('numbers of cluster')
# plt.ylabel('inertia_')
# plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters=5,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
km.fit(X_part)

X['cluster_result'] = km.predict(X_part)

del X['radius error']
del X['compactness error']
del X['concavity error']

print(X.info())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0,
                           class_weight='balanced',
                           random_state=0)
model.fit(X_train_scaled, y_train)

v_score = model.score(X_train_scaled, y_train)
print(f'학습 : {v_score}')
v_score = model.score(X_test_scaled, y_test)
print(f'테스트 : {v_score}')

# 선형 모델은 각각의 특성들에 대해서 가중치를 계산하는 모델
# 가중치가 클수록 중요도가 높음
print(f'학습된 가중치 : \n{model.coef_}')

# 가중치가 적은 10, 15, 16
print(X.info())
['radius error', 'compactness error', 'concavity error']



























