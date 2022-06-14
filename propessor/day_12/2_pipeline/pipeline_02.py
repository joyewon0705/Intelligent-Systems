# -*- coding: utf-8 -*-

# 2_pipeline.py

# 일반적인 머신러닝 단계
# - 데이터 전처리 단계의 추가

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
scaler=StandardScaler().fit(X_train)

X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

print(X_train[:5])
print(X_train_scaled[:5])

# 3. 머신러닝 모델 객체의 생성과 학습
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(
    C=1.0, n_jobs=-1, 
    random_state=11).fit(X_train_scaled,y_train)

# 4. 학습된 머신러닝 모델의 평가
print(f'SCORE(TRAIN) : {model.score(X_train_scaled, y_train)}')
print(f'SCORE(TEST) : {model.score(X_test_scaled, y_test)}')

from sklearn.metrics import classification_report
pred_train=model.predict(X_train_scaled)
pred_test=model.predict(X_test_scaled)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))











