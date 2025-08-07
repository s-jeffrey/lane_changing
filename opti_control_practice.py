import random
import numpy as np
from deap import base, creator, tools, algorithms

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_float", random.uniform, -10.0, 10.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    x, y = individual
    return ((x - 3)**2 + (y - 2)**2),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population_size = 50
    number_generations = 50
    crossover_probability = 0.7
    mutation_probability = 0.2

    pop = toolbox.population(n=population_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        ngen=number_generations,
        stats=stats,
        verbose=True
    )

    best_individual = tools.selBest(pop, 1)[0]
    
    print(f"Best individual is {best_individual}, with fitness {best_individual.fitness.values[0]}")

    return pop, logbook, best_individual

if __name__ == "__main__":
    main()