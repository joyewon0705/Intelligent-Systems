# -*- coding: utf-8 -*-

# 4_pipeline.py

# 일반적인 머신러닝 단계
# - 데이터 전처리 단계의 추가
# - 하이퍼 파라메터 검색 단계의 추가
# - 파이프 라인을 사용한 데이터의 전처리 과정 및 
#   머신러닝 모델의 학습 과정 자동화 (전처리와 학습의 연결)

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

# 3. 머신러닝 모델 객체의 생성과 학습
# - 하이퍼 파라메터 검색을 통한 최적의 모델을 생성
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 파이프 라인 객체의 생성
# - Pipeline([(1번째 변환기 클래스 객체의 이름, 객체), (2번째 변환기 클래스 객체의 이름, 객체), ...])
# - Pipeline([1번째 변환기 클래스의 튜플, 2번째 변환기 클래스의 튜플, ...])
# - 파이프 라인의 마지막 객체를 제외한 나머지 객체들은
#   transform, fit_transform 메소드를 제공하는 변환기만 허용
# - 파이프 라인의 마지막 객체는 predict 메소드를 제공하는 
#   예측기 객체가 될 수 있음

base_model = LogisticRegression(n_jobs=-1,random_state=11)

pipe = Pipeline([('s_scaler',scaler),('base_model',base_model)])

# 파이프 라인의 실행 과정
# 1. fit 메소드가 호출되는 경우
# - 입력된 X 데이터를 첫번째 변환기 클래스의 객체로 전달(fit 메소드가 호출)
# - 입력된 X 데이터를 transform 메소드를 통해서 변환 시킴
# - 변환된 입력 데이터 X를 다음에 위치한 변환기 또는 예측기로 전달하여
#   fit 메소드를 실행
# - 다음의 객체가 변환기 객체인 경우 transform 메소드의 실행 결과를 반환하여
#   다음 객체로 전달하고, 예측기 클래스인 경우 실행을 종료

pipe.fit(X_train,y_train)

# 2. score/predict 메소드가 호출되는 경우
# - 입력된 X 데이터를 첫번째 변환기 클래스의 객체로 전달(transform 메소드가 호출)
# - 변환된 X 데이터를 다음의 객체로 전달
# - 다음의 객체가 변환기인 경우 다시 한번 transform 메소드가 호출되어 변환된 결과를
#   다음의 객체에게 전달하고
#   만약 예측기 객체인 경우 변환된 X 데이터를 사용하여 score/predict 메소드의
#   실행 결과가 반환

# 4. 학습된 머신러닝 모델의 평가
print(f'SCORE(TRAIN) : {pipe.score(X_train, y_train)}')
print(f'SCORE(TEST) : {pipe.score(X_test, y_test)}')

from sklearn.metrics import classification_report
pred_train=pipe.predict(X_train)
pred_test=pipe.predict(X_test)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))











