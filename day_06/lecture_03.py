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
# - 랜덤포레스트 : 배깅 방법론에 결정 트리를 조합하여 사용하는
#                 패턴이 빈번하게 발생하여 해당 구조를 하나의
#                 앙상블 모형으로 구현해 놓은 클래스
# - 배깅 + 결정트리
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,
                               max_depth=None,
                               max_samples=0.3,
                               max_features=0.5,
                               n_jobs=-1,
                               random_state=1)

model.fit(X_train,  y_train)


score = model.score(X_train, y_train)
print(f'Score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score (Test) : {score}')







