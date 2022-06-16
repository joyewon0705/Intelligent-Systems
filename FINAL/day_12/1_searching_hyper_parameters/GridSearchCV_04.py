# -*- coding: utf-8 -*-

# GridSearchCV_04.py

# 머신러닝 모델의 하이퍼 파라메터를
# 검색하는 예제

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 교차검증 데이터 셋을 분할하기 위한 클래스
from sklearn.model_selection import KFold
# 교차검증을 수행할 수 있는 함수
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

# 전체 데이터를 훈련 및 테스트 세트로 분할
X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y)

# 모델의 학습에 사용할 파라메터의 정의
param_grid = {'learning_rate':[0.1, 0.2, 0.3, 1., 0.01],
              'max_depth':[1, 2, 3],
              'n_estimators':[100, 200, 300, 10, 50]}

# 교차 검증 점수를 기반으로 최적의 하이퍼 파라메터를 
# 검색할 수 있는 GridSearchCV 클래스
from sklearn.model_selection import GridSearchCV

# 교차 검증 수행을 위한 데이터 분할 객체
cv=KFold(n_splits=5,shuffle=True,random_state=1)
# 교차 검증에 사용할 기본 머신러닝 모델
# - 테스트할 하이퍼 파라메러는 설정에서 제외
# - 공통 하이퍼 파라메터 정보만 사용하여 모델을 정의
# - 학습 X
base_model=GradientBoostingClassifier(random_state=1)

# GridSearchCV 클래스의 하이퍼 파라메터 정보
# GridSearchCV(
#   예측기 객체, 
#   테스트 파라메터의 딕셔너리 객체, 
#   cv=교차검증 폴드 수,...)
grid_model = GridSearchCV(estimator=base_model,
                          param_grid=param_grid,
                          cv=cv,
                          n_jobs=-1)
grid_model.fit(X_train,y_train)

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








