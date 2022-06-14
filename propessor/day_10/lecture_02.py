# -*- coding: utf-8 -*-

import pandas as pd

X = pd.DataFrame()

print(X)
print(X.info())

X['gender'] = ['F','M','F','F','M']
print(X)

X['age'] = [15, 35, 25, 37, 55]
print(X)

# 데이터 전처리
# 1. 문자열
# - 결측 데이터
# - 라벨 인코딩
# - 원핫 인코딩

# 2. 수치형
# - 결측 데이터
# - 이상치 제거(대체)
# - 스케일링

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

encoder = OneHotEncoder(sparse=False,
                        handle_unknown='ignore')

scaler = MinMaxScaler()

from sklearn.compose import ColumnTransformer

obj_columns=['gender']
num_columns=['age']

ct = ColumnTransformer(
    [('scaler',scaler,num_columns),
     ('encoder',encoder,obj_columns)])

ct.fit(X)

print(X)
print(ct.transform(X))








