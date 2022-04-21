# -*- coding: utf-8 -*-

# 앙상블 (Ensemble)
# - 다수개의 머신러닝 알고리즘을 결합하여
#   각 모델이 예측한 결과를 취합/부스팅 방법을 통해
#   예측을 수행하는 방법 / 방법론

# 앙상블의 구현 방식
# 1. 취합
# - 앙상블을 구성하는 내부의 각 모델이 서로 독립적으로 동작
# - 각각의 모델이 예측한 결과값에 대해서 다수결 방식을 수행(분류)
# - 각각의 모델이 예측한 결과값에 대해서 평균을 취함(회귀분석)
# - 내부의 각 모델은 서로 연관성이 존재하지 않음
# - 취합 방식의 앙상블 모델을 구축하는 경우 내부의 각 모델은
#   적절한 수준으로 과적합을 수행할 필요가 있음
# - 학습 / 예측의 수행 속도가 빠름
#   (각 모델이 독립적이므로 병렬 처리가 가능함)
# - Voting, Bagging, RandomForest

# 2. 부스팅
# - 앙상블을 구성하는 내부의 각 모델들이 선형으로 연결되어
#   학습 및 예측을 수행하는 방식
# - 내부의 각각의 모델은 다음의 모델에 영향을 주는 방식
# - 부스팅 방식의 앙상블 모델은 내부의 각 모델들에 대해서
#    강한 제약을 설정하여 점진적인 성능향상을 도모해야함
# - 학습 / 예측의 수행 속도가 느림
#   (각 모델은 선형으로 연결되어 앞의 모델이 학습이 종료된 후
#    이후의 모델이 학습되는 구조이므로)
# - AdaBoosting, GradientBoosting, XGBoost, LightGBM

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

y.head()
y.tail()
y.value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

# 앙상블 클래스의 로딩
from sklearn.ensemble import VotingClassifier

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

m1 = KNeighborsClassifier(n_jobs=-1)
m2 = LogisticRegression(n_jobs=-1, random_state=1)
m3 = DecisionTreeClassifier(random_state=1)

estimators = [('knn', m1), ('lr', m2), ('dt', m3)]

model = VotingClassifier(estimators=estimators,
                         voting='hard',
                         n_jobs=-1)

model.fit(X_train,  y_train)

score = model.score(X_train, y_train)
print(f'Score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score (Test) : {score}')


# 앙상블 모델의 결과
pred = model.predict(X_test[50:51])
print(f'Predict : {pred}')

# 앙상블 내부의 구성 모델 확인
print(model.estimators_[0])
print(model.estimators_[1])
print(model.estimators_[2])

# 앙상블 내부의 각 모델의 예측 값 확인
pred = model.estimators_[0].predict(X_test[50:51])
print(f'Predict (knn) : {pred}')

pred = model.estimators_[1].predict(X_test[50:51])
print(f'Predict (lr) : {pred}')

pred = model.estimators_[0].predict(X_test[50:51])
print(f'Predict (dt) : {pred}')

















