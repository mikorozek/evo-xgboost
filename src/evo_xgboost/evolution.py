import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score

from .settings import POPULATION_SIZE, NUMBER_OF_GENERATIONS, ELITE_SIZE, PARAM_RANGES
from .individual import create_individual, calculate_fitness
from .operators import tournament_selection, crossover, mutate

def run_evolution(X_train, y_train, X_test, y_test):
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_overall_fitness = -1
    best_overall_individual = None

    for generation in range(NUMBER_OF_GENERATIONS):
        print(f"Generation {generation + 1}/{NUMBER_OF_GENERATIONS}")

        fitnesses = [calculate_fitness(ind, X_train, y_train, X_test, y_test) for ind in population]

        best_gen_fitness = max(fitnesses)
        if best_gen_fitness > best_overall_fitness:
            best_overall_fitness = best_gen_fitness
            best_overall_individual = population[np.argmax(fitnesses)]

        print(f"Best fitness in generation {generation + 1}: {best_gen_fitness:.4f}")
        print(f"Best overall fitness: {best_overall_fitness:.4f}")

        elites = [population[i] for i in np.argsort(fitnesses)[-ELITE_SIZE:]]

        parents = tournament_selection(population, fitnesses)
        
        offspring = []
        if len(parents) % 2 != 0:
            parents.pop()

        for i in range(0, len(parents), 2):
            child1, child2 = crossover(parents[i], parents[i+1])
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))
        
        population = elites + offspring
        while len(population) < POPULATION_SIZE:
            population.append(create_individual())
        population = population[:POPULATION_SIZE]

    print("\nEvolution finished.")
    print("Best individual found:")
    print(best_overall_individual)
    print(f"Best F1-score: {best_overall_fitness:.4f}")

    return best_overall_individual
