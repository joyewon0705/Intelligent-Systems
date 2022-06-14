# -*- coding: utf-8 -*-

# 1_pipeline.py

# 일반적인 머신러닝 단계

# 1. 데이터의 적재 및 분할
from sklearn.datasets import load_breast_cancer

# - 데이터 적재
X, y = load_breast_cancer(return_X_y=True)

# - 데이터 분할
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=11)

# 2. 머신러닝 모델 객체의 생성과 학습
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(
    C=1.0, n_jobs=-1, 
    random_state=11).fit(X_train,y_train)

# 3. 학습된 머신러닝 모델의 평가
print(f'SCORE(TRAIN) : {model.score(X_train, y_train)}')
print(f'SCORE(TEST) : {model.score(X_test, y_test)}')

from sklearn.metrics import classification_report
pred_train=model.predict(X_train)
pred_test=model.predict(X_test)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))











