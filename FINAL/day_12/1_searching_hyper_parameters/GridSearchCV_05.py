# -*- coding: utf-8 -*-

# GridSearchCV_06.py

# 머신러닝 모델의 하이퍼 파라메터를
# 검색하는 예제

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y)

# 데이터의 스케일 정보를 확인
# - 전처리 여부를 판단
import pandas as pd
X_df = pd.DataFrame(X)
# 스케일의 편차가 존재하기 때문에
# 정규화 처리가 필요함
print(X_df.describe(include='all'))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 모델의 학습에 사용할 파라메터의 정의
# - LogisticRegression

# - 아래의 파라메터 정의는 solver와 penalty 값의 
#   조합에 따라 성공, 실패가 될 수 있습니다.
#   (elasticnet은 saga만 지원합니다.)
# param_grid = {'C':[1., 0.1, 0.01, 10., 100],
#              'penalty':['l2', 'elasticnet'],
#              'solver':['lbfgs', 'liblinear', 'sag', 'saga']}

# 조건부 매개변수를 사용하기 위한 매개변수 그리드 선언
# 사용 방식 
# - [{조건부 매개변수 1}, {조건부 매개변수 2} ... ]
# - 아래의 매개변수 그리드는 
#   1번째 l1 penalty에 대한 매개변수 그리드
#   2번째 l2 penalty에 대한 매개변수 그리드
#   3번째 elasticnet penalty에 대한 매개변수 그리드
param_grid = [{'C':[1., 0.1, 0.01, 10., 100],
              'penalty':['l1'],
              'solver':['liblinear', 'saga']},
              {'C':[1., 0.1, 0.01, 10., 100],
              'penalty':['l2'],
              'solver':['lbfgs', 'sag', 'saga']},
              {'C':[1., 0.1, 0.01, 10., 100],
              'penalty':['elasticnet'],
              'solver':['saga']}]

from sklearn.model_selection import GridSearchCV
cv=KFold(n_splits=5,shuffle=True,random_state=1)
estimator=LogisticRegression(max_iter=10000)

grid_model = GridSearchCV(estimator=estimator,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1).fit(X_train,y_train)

# 모든 하이퍼 파라메터를 조합하여 평가한 
# 가장 높은 교차검증 SCORE 값을 반환
print(f'best_score -> {grid_model.best_score_}')
# 가장 높은 교차검증 SCORE 가 어떤 
# 하이퍼 파라메터를 조합했을 때 만들어 졌는지 확인
print(f'best_params -> {grid_model.best_params_}')
# 가장 높은 교차검증 SCORE의 
# 하이퍼 파라메터를 사용하여 생성된 모델 객체를 반환
print(f'best_model -> {grid_model.best_estimator_}')

score = grid_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = grid_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')








