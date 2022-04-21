# -*- coding: utf-8 -*-

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_diabetes

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.head()
X.info()
X.describe()

y.head()
y.tail()
y.value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

# 앙상블 클래스의 로딩
# - 부스팅 계열
# - 앙상블을 구현하는 내부의 각 모델들이 선형으로 연결되어
#   학습 및 예측을 수행하는 방법론

# 1. AdaBoost : 데이터 중심의 부스팅 방법론을 구현
#               (직전 모델이 잘못 예측한 데이터에
#                가중치를 부여하는 방법)

# 2. GradientBoosting : 오차에 중심을 둔 부스팅 방법론을 구현
#                       (각각의 학습 데이터에 대해서 오차의 크기가
#                        큰 데이터에 가중치를 부여하여 전체 
#                        오차를 줄여나가는 방식)

# 무스팅 계열의 데이터 예측 예시
# 1번째 모델의 예측값 * 가중치(1번째 모델의 가중치) +
# 2번째 모델의 예측값 * 가중치(2번째 모델의 가중치) +
# ...
# N번째 모델의 예측값 * 가중치(N번째 모델의 가중치)

# GradientBoosting 클래스는 부스팅을 구현하기 위한
# 기본 모델이 결정트리로 고정되어 있음
# - 랜덤포레스트의 부스팅 버전!
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=50,
                                   learning_rate=0.1,
                                   max_depth=1,
                                   subsample=0.3,
                                   max_features=0.3,
                                   random_state=1,
                                   verbose=3)

model.fit(X_train,  y_train)


score = model.score(X_train, y_train)
print(f'Score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score (Test) : {score}')







