# -*- coding: utf-8 -*-

# GridSearchCV_03.py

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

# 최적의 하이퍼 파라메터를 검색하는 코드
best_score = None
best_params = {}

for lr in [0.1, 0.2, 0.3, 1., 0.01] :
    for md in [1, 2, 3] :
        for ne in [100, 200, 300, 10, 50] :
            
            # 각 하이퍼 파라메터의 조합을 사용하여
            # 모델을 생성하고 학습
            # - 학습 데이터(X_train)를 사용하여 fitting!!!
            model = GradientBoostingClassifier(
                        learning_rate=lr,
                        max_depth=md,
                        n_estimators=ne,
                        random_state=1).fit(X_train, y_train)

            # 교차 검증을 사용하여 학습데이터에 대한
            # 모델의 성능을 평가
            cv=KFold(n_splits=5,shuffle=True,random_state=1)
            scores = cross_val_score(model,
                                     X_train,y_train,
                                     cv=cv,
                                     n_jobs=-1)
            
            # 교차 검증에 대한 정확도의 평균을 계산
            score = np.mean(scores)
            
            # score 점수를 기반으로 
            # 최적의 모델 정보를 저장
            if not best_score or \
                (best_score and score > best_score) :                
                best_score = score
                best_params = {'learning_rate' : lr,
                               'max_depth' : md,
                               'n_estimators' : ne,
                               'random_state' : 1}
                                
print(f'best_score : {best_score}')
print(f'best_params : \n{best_params}')

# 가장 높은 score 를 기록한 하이퍼 파라메터를 사용하여
# 모델을 생성한 후, 테스트 데이터를 사용해 평가
best_model = GradientBoostingClassifier(**best_params).fit(X_train, y_train)

score = best_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = best_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')











