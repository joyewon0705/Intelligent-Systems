# -*- coding: utf-8 -*-

# GridSearchCV_02.py

# 머신러닝 모델의 하이퍼 파라메터를
# 검색하는 예제

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

# 학습 데이터를 분할하여 검증 데이터 셋을 추가로 생성
X_train,X_valid,y_train,y_valid=train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=1,
    stratify=y_train)

# 학습 / 검증 / 테스트 데이터 셋의 크기를 확인
# TRAIN : 108, VALID : 12, TEST : 30
print(f'TRAIN : {len(X_train)}, VALID : {len(X_valid)}, TEST : {len(X_test)}')

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

            # 학습된 모델을 사용하여 
            # 검증 데이터의 성능을 평가
            score = model.score(X_valid, y_valid)
            # 문제 : 데이터가 어떻게 분할되느냐에 따라 학습의 성능이 달라짐
            
            print(score, '0')
            # score 점수를 기반으로 
            # 최적의 모델 정보를 저장
            if not best_score or \
                (best_score and score > best_score) :
                print(score, '1')
                best_score = score
                best_params = {'learning_rate' : lr,
                               'max_depth' : md,
                               'n_estimators' : ne,
                               'random_state' : 1}
                                
print(f'best_score : {best_score}')
print(f'best_params : \n{best_params}')

# 가장 높은 score 를 기록한 하이퍼 파라메터를 사용하여
# 모델을 생성한 후 테스트 데이터를 사용해 평가
best_model = GradientBoostingClassifier(
                        learning_rate=best_params['learning_rate'],
                        max_depth=best_params['max_depth'],
                        n_estimators=best_params['n_estimators'],
                        random_state=1).fit(X_train, y_train)

score = best_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = best_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')











