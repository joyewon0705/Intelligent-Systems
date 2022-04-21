# -*- coding: utf-8 -*-

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.head()

y.head()
y.tail()
y.value_counts()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

# 앙상블 클래스의 로딩
# - 배깅 : 특정 머신러닝 알고리즘을 기반으로
#          데이터의 무작위 추출을 사용하여
#          각 모델들이 서로 다른 데이터를 학습하는 방식으로
#          앙상블을 구현하는 방법
from sklearn.ensemble import BaggingClassifier

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩
from sklearn.tree import DecisionTreeClassifier

base_estimator = DecisionTreeClassifier(max_depth=3, random_state=1)

model = BaggingClassifier(base_estimator=base_estimator,
                          n_estimators=50,
                          max_samples=0.2,
                          max_features=0.2,
                          n_jobs=-1,
                          random_state=1)

model.fit(X_train,  y_train)


score = model.score(X_train, y_train)
print(f'Score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score (Test) : {score}')

# 공부의 성적을 올리고 싶다?
# - samples, features를 최대치(1.0)으로 설정
# - max_depth 제약 X










