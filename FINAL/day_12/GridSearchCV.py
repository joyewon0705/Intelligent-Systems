# -*- coding: utf-8 -*-

# GridSearchCV_04.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

param_grid = {'learning_rate':[0.1, 0.2, 0.3, 1., 0.01],
              'max_depth':[1, 2, 3],
              'n_estimators':[100, 200, 300, 10, 50]}

from sklearn.model_selection import GridSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=1)

base_model = GradientBoostingClassifier(random_state=1)

grid_model  = GridSearchCV(estimator=base_model, 
                           param_grid=param_grid,
                           cv=cv,
                           n_jobs=-1)

grid_model.fit(X_train, y_train)

print(f'best_score -> {grid_model.best_score_}')
print(f'best_params -> {grid_model.best_params_}')
print(f'best_estimators -> {grid_model.best_estimator_}')

score = grid_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = grid_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')















