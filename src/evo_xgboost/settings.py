PARAM_RANGES = {
    'n_estimators': ('discrete', 50, 500),
    'eta': ('continuous', 0.01, 0.3),
    'max_depth': ('discrete', 3, 10),
    'subsample': ('continuous', 0.5, 1.0),
    'colsample_bytree': ('continuous', 0.5, 1.0),
    'min_child_weight': ('discrete', 1, 10),
    'gamma': ('continuous', 0.0, 0.5)
}

POPULATION_SIZE = 20
NUMBER_OF_GENERATIONS = 50
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3
ELITE_SIZE = 2

DATA_URL = 'https://calmcode.io/static/data/titanic.csv'
