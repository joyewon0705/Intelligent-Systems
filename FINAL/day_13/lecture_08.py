# -*- coding: utf-8 -*-

# 스태킹 모델의 구축
# - 앙상블 : 다수개의 머신러닝 모델의 예측 값을 취합하여
#   평균, 다수결의 원칙으로 예측하는 모델
# - 앙상블을 사용하는 이유???
#   일반화의 성능을 극대화하기 위해서
#   (예측 성능의 분산을 감소시킬 수 있으므로)

# - 다수개의 머신러닝 모델이 예측한 값을 학습하여
#   결과를 반환하는 방법 -> 스태킹
# - 머신러닝 알고리즘에 의해서 원본 학습데이터가
#   하나의 결과 값으로 압축되어 머신러닝의 개수만큼의
#   데이터로 생성되고 해당 데이터를 학습하여 예측하는 방식

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(X.head())
print(X.info())
print(X.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 앙상블의 구현
# - 다수개의 모델을 사용하여 학습 및 예측을 진행
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

clf_lr = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
clf_kn = KNeighborsClassifier().fit(X_train_scaled, y_train)
clf_dt = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, y_train)

v_score = clf_lr.score(X_train_scaled, y_train)
print(f'학습 clf_lr : {v_score}')
v_score = clf_kn.score(X_train_scaled, y_train)
print(f'학습 clf_kn : {v_score}')
v_score = clf_dt.score(X_train_scaled, y_train)
print(f'학습 clf_ddt : {v_score}')

v_score = clf_lr.score(X_test_scaled, y_test)
print(f'테스트 clf_lr : {v_score}')
v_score = clf_kn.score(X_test_scaled, y_test)
print(f'테스트 clf_kn : {v_score}')
v_score = clf_dt.score(X_test_scaled, y_test)
print(f'테스트 clf_ddt : {v_score}')

# 스태킹 구현
# 1. 앙상블을 구현하고 있는 각 머신러닝 모델들의 예측 결과를 취합
pred_lr = clf_lr.predict(X_train_scaled)
pred_kn = clf_kn.predict(X_train_scaled)
pred_dt = clf_dt.predict(X_train_scaled)

import numpy as np
pred_stack = np.array([pred_lr, pred_kn, pred_dt])
# print(pred_stack)

# print(y_train.shape)
# print(pred_stack.shape)

pred_stack = pred_stack.T

# print(pred_stack.shape)
# print(pred_stack[:5])

from sklearn.ensemble import RandomForestClassifier
final_model = RandomForestClassifier(n_estimators=100,
                                     max_depth=None,
                                     max_samples=0.5,
                                     max_features=0.3,
                                     random_state=1).fit(pred_stack, y_train)
score = final_model.score(pred_stack, y_train)
print(f'학습 final_model : {score}')

# 평가
# - 테스트 데이터에 대한 각 모델들의 예측값을 취합
pred_lr = clf_lr.predict(X_test_scaled)
pred_kn = clf_kn.predict(X_test_scaled)
pred_dt = clf_dt.predict(X_test_scaled)

pred_stack = np.array([pred_lr, pred_kn, pred_dt]).T

score = final_model.score(pred_stack, y_test)
print(f'테스트 final_model : {score}')





