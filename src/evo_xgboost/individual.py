import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

from .settings import PARAM_RANGES

def create_individual():
    individual = {}
    for param, (ptype, pmin, pmax) in PARAM_RANGES.items():
        if ptype == 'continuous':
            individual[param] = np.random.uniform(pmin, pmax)
        elif ptype == 'discrete':
            individual[param] = np.random.randint(pmin, pmax + 1)
    return individual

def calculate_fitness(individual, X_train, y_train, X_test, y_test):
    params = individual.copy()
    for param, value in params.items():
        if PARAM_RANGES[param][0] == 'discrete':
            params[param] = int(value)
    
    model = xgb.XGBClassifier(
        **params,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)
