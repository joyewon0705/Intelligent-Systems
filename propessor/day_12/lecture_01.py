# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(X.head())
print(X.info())
print(X.describe())

print(y.head())
print(y.describe())


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=\
    train_test_split(X, y,
                     test_size=0.3,                     
                     random_state=11)


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

s_mm = MinMaxScaler()
s_ss = StandardScaler()
s_rs = RobustScaler()

mm_cols=['MedInc','HouseAge','AveRooms','AveBedrms']
ss_cols=['AveOccup','Latitude','Longitude']
rs_cols=['Population']

from sklearn.compose import ColumnTransformer
pp = ColumnTransformer([
    ('s_mm', s_mm, mm_cols),
    ('s_ss', s_ss, ss_cols),
    ('s_rs', s_rs, rs_cols)],
    n_jobs=-1)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1, 
                              random_state=1)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('pp',pp),
                 ('model',model)])

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

param_grid = {'model__n_estimators':[100, 50, 10, 200],
              'model__max_depth':[None, 7, 10]}

cv = KFold(n_splits=15, shuffle=True, random_state=1)
grid = GridSearchCV(pipe, 
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=1,
                    verbose=3,
                    scoring='r2')

grid.fit(X_train, y_train)

print(f'best_score : {grid.best_score_}')
print(f'best_params : {grid.best_params_}')
print(f'best_model : {grid.best_estimator_}')


# 4. 학습된 머신러닝 모델의 평가
print(f'SCORE(TRAIN) : {grid.score(X_train, y_train)}')
print(f'SCORE(TEST) : {grid.score(X_test, y_test)}')




