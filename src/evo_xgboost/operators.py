import numpy as np
from .settings import PARAM_RANGES, CROSSOVER_PROB, MUTATION_PROB, TOURNAMENT_SIZE

def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_index])
    return selected

def crossover(parent1, parent2):
    if np.random.rand() < CROSSOVER_PROB:
        child1, child2 = {}, {}
        for param in parent1:
            if np.random.rand() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        return child1, child2
    return parent1, parent2

def mutate(individual):
    for param, (ptype, pmin, pmax) in PARAM_RANGES.items():
        if np.random.rand() < MUTATION_PROB:
            if ptype == 'continuous':
                new_val = individual[param] + np.random.normal(0, 0.1 * (pmax - pmin))
                individual[param] = np.clip(new_val, pmin, pmax)
            elif ptype == 'discrete':
                new_val = individual[param] + np.random.normal(0, 1)
                individual[param] = int(np.clip(round(new_val), pmin, pmax))
    return individual
