# -*- coding: utf-8 -*-

# 8_searching_hyper_parameters
# GridSearchCV_01.py

# 머신러닝 모델의 하이퍼 파라메터를
# 검색하는 예제

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y)

# 최적의 하이퍼 파라메터를 검색하는 코드
best_score = None
best_params = {}

# learning_rate 파라메터의 값을 제어
for lr in [0.1, 0.2, 0.3, 1., 0.01] :
    # max_depth 파라메터의 값을 제어
    for md in [1, 2, 3] :
        # n_estimators 파라메터의 값을 제어
        for ne in [100, 200, 300, 10, 50] :
            
            # 각 하이퍼 파라메터의 조합을 사용하여
            # 모델을 생성하고 학습
            model = GradientBoostingClassifier(
                        learning_rate=lr,
                        max_depth=md,
                        n_estimators=ne,
                        random_state=1).fit(X_train, y_train)

            # 학습된 모델을 사용하여 
            # 테스트 데이터의 성능을 평가
            score = model.score(X_test, y_test)
            
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







