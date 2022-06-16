# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.info())

print(X.describe())


# 데이터 전처리
# 1. 문자열
# - 결측 데이터
# - 라벨 인코딩
# - 원핫 인코딩

# 2. 수치형
# - 결측 데이터
# - 이상치 제거(대체)
# - 스케일링

# - 전처리 클래스 로딩
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# - 전처리를 적용할 컬럼을 식별
num_columns = X.columns

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('scaler', scaler, num_columns)])
ct.fit(X)

print(X.head())
print(ct.transform(X)[:5])























