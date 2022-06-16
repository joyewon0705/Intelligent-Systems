# -*- coding: utf-8 -*-

# 5_pipeline.py

# 일반적인 머신러닝 단계
# - 데이터 전처리 단계의 추가
# - 하이퍼 파라메터 검색 단계의 추가
# - 파이프 라인을 사용한 데이터의 전처리 과정 및 
#   머신러닝 모델의 학습 과정 자동화 (전처리와 학습의 연결)
# - 파이프 라인을 사용한 하이퍼 파라메터를 검색
#   (올바른 방식의 교차 검증을 수행)

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer

# - 데이터 적재
X, y = load_breast_cancer(return_X_y=True)

# - 데이터의 확인
#   (결측 데이터 여부, 스케일 확인)
import pandas as pd
pd.options.display.max_columns=100
X_df=pd.DataFrame(X)

print(X_df.info())
print(X_df.describe())

# - 데이터 분할
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=11)

# 2. 데이터의 전처리 과정    
# - 라벨 인코딩, 특성 데이터의 스케일 조정 등의 작업을 수행
# - 사이킷 런의 변환기 클래스를 활용
# - fit 메소드는 반드시 학습(TRAIN) 데이터에 대해서만 적용
# - transform 메소드를 사용하여 학습 및 테스트 데이터의 변환을 수행
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 3. 하이퍼 파라메터 검색을 통한 최적의 모델 생성
# - 파이프 라인을 사용하여 데이터 전처리 및 
#   머신러닝 모델의 하이퍼 파라메터 검색
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pipeline을 예측기로 사용하는 GridSearchCV 클래스의
# 파라메터 정보는 키 값의 형태를 
# 파이프라인의예측기객체명__파라메터이름
param_grid=[{'base_model__penalty' : ['l2'],
            'base_model__solver' : ['lbfgs'],
            'base_model__C' : [1.0,0.1,10,0.01,100],
            'base_model__class_weight': ['balanced',{0:0.9,1:0.1}]},
            {'base_model__penalty' : ['elasticnet'],
            'base_model__solver' : ['saga'],
            'base_model__C' : [1.0,0.1,10,0.01,100],
            'base_model__class_weight': ['balanced',{0:0.9,1:0.1}]}]

cv = KFold(n_splits=5,shuffle=True,random_state=11)

base_model = LogisticRegression(n_jobs=-1,random_state=11)

pipe = Pipeline([('s_scaler',scaler),('base_model',base_model)])

# GridSearchCV 클래스의 생성자 매개변수로 
# 파이프 라인 객체가 사용될 수 있습니다.
# - 아래의 예는 폴드가 5개로 지정되어 4개의 폴드를 사용하여
#   데이터 정규화를 처리한 후 학습을 진행합니다.
# - 남은 하나의 폴드는 기존의 4개의 폴드로 전처리된 변환기 클래스에
#   의해서 transform 되어 예측에 사용됩니다.
#   (새로운 데이터로 인식되는 방식)
grid_model = GridSearchCV(pipe, param_grid=param_grid, cv=cv,
                          scoring='f1', n_jobs=-1)

grid_model.fit(X_train,y_train)

print(f'best_score : {grid_model.best_score_}')
print(f'best_params : {grid_model.best_params_}')
print(f'best_model : {grid_model.best_estimator_}')


# 4. 학습된 머신러닝 모델의 평가
print(f'SCORE(TRAIN) : {grid_model.score(X_train, y_train)}')
print(f'SCORE(TEST) : {grid_model.score(X_test, y_test)}')

from sklearn.metrics import classification_report
pred_train=grid_model.predict(X_train)
pred_test=grid_model.predict(X_test)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))














