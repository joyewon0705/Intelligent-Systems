# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv('./titanic.csv', header='infer', sep=',')
data.info()

X = data.drop(['PassengerId','Cabin','Name','Ticket'], axis=1)
y = data['Survived']

obj_col = [cname for cname in X.columns if X[cname].dtype == 'object']
num_col = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]

X.info()
X.isnull().sum()

from sklearn.impute import SimpleImputer
obj_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer()

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
scaler = MinMaxScaler()
encoder = OneHotEncoder()

from sklearn.pipeline import Pipeline
obj_pp = Pipeline([('obj_imputer', obj_imputer),
                   ('encoder', encoder)])
num_pp = Pipeline([('num_imputer', num_imputer),
                   ('scaler', scaler)])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('obj_pp', obj_pp, obj_col),
                        ('num_pp', num_pp, num_col)])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=1)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
model_gb = GradientBoostingClassifier(random_state=1)
model_rf = RandomForestClassifier(random_state=1)
model_svc = SVC()

pp_gb = Pipeline([('ct', ct),
                  ('model_gb', model_gb)]).fit(X_train, y_train)
pp_rf = Pipeline([('ct', ct),
                  ('model_rf', model_rf)]).fit(X_train, y_train)
pp_svc = Pipeline([('ct', ct),
                   ('model_svc', model_svc)]).fit(X_train, y_train)

pred_gb = pp_gb.predict(X_train)
pred_rf = pp_rf.predict(X_train)
pred_svc = pp_svc.predict(X_train)

import numpy as np
pred_stack = np.array([pred_gb, pred_rf, pred_svc]).T


from sklearn.ensemble import RandomForestClassifier
final_model = RandomForestClassifier(n_estimators=100,
                                     max_depth=None,
                                     max_samples=0.5,
                                     max_features=0.3,
                                     random_state=1).fit(pred_stack, y_train)

score = final_model.score(pred_stack, y_train)
print(f'학습 final_model : {score}')

pred_gb = pp_gb.predict(X_test)
pred_rf = pp_rf.predict(X_test)
pred_svc = pp_svc.predict(X_test)

pred_stack = np.array([pred_gb, pred_rf, pred_svc]).T

score = final_model.score(pred_stack, y_test)
print(f'테스트 final_model : {score}')






