# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

base_model = LogisticRegression(n_jobs=-1, random_state=11)

pipe = Pipeline([('scaler', scaler), 
                 ('base_model', base_model)])

param_grid = [{'base_model__penalty' : ['l2'],
               'base_model__solver' : ['lbfgs'],
               'base_model__C' : [1.0,0.1,10,0.01,100],
               'base_model__class_weight' : ['balanced', {0:0.9,1:0.1}]},
              {'base_model__penalty' : ['elasticnet'],
               'base_model__solver' : ['saga'],
               'base_model__C' : [1.0,0.1,10,0.01,100],
               'base_model__class_weight' : ['balanced', {0:0.9,1:0.1}]}]

cv = KFold(n_splits=5, shuffle=True, random_state=11)

grid_model = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          cv=cv,
                          scoring='recall', # 재현율에 집중해서 공부해봐라
                          n_jobs=-1)

grid_model.fit(X_train, y_train)

print(f'best_score : {grid_model.best_score_}')
print(f'best_params : {grid_model.best_params_}')
print(f'best_model : {grid_model.best_estimator_}')

from sklearn.metrics import classification_report

classification_report(y_train, grid_model.predict(X_train))
classification_report(y_test, grid_model.predict(X_test))
