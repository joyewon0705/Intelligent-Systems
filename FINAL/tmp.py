# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=11)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
s_mm = MinMaxScaler()
s_ss = StandardScaler()
s_rs = RobustScaler()

mm_cols = ['MedInc','HouseAge','AveRooms','AveBedrms']
ss_cols = ['AveOccup','Latitude','Longitude']
rs_cols = ['Population']

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('s_mm', s_mm, mm_cols),
                        ('s_ss', s_ss, ss_cols),
                        ('s_rs', s_rs, rs_cols)],
                       n_jobs=-1)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1, random_state=11)

from sklearn.pipeline import Pipeline
pp = Pipeline([('ct', ct), ('model', model)])

from sklearn.model_selection import KFold, GridSearchCV
param_grid = {'model__n_estimators' : [100, 50, 10, 200],
              'model__max_depth' : [None, 7, 10]}

cv = KFold(n_splits=15, shuffle=True, random_state=11)

grid = GridSearchCV(estimator=pp, 
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=-1)

grid.fit(X_train, y_train)

