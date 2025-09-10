import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from deap import base, creator, tools, algorithms

x0 = 5.0
xtarget = 0.0

N_CTRL_POINTS = 100

PENALTY = 50.0

# GA parameters
POPULATION_SIZE = 1000
CXPB = 0.7  # Crossover probability
MUTPB = 0.4 # Mutation probability

# DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def gen_individual():
    u_values = [random.uniform(-1.0, 1.0) for _ in range(N_CTRL_POINTS)]
    T = random.uniform(abs(x0), 2 * abs(x0))
    return creator.Individual(u_values + [T])

toolbox.register("individual", gen_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    u_sequence = individual[:-1]
    T = individual[-1]

    if T <= 0:
        return (float('inf'),)

    dt = T / N_CTRL_POINTS
    x = x0
    for index, u in enumerate(u_sequence):
        x += u * dt

        # if x <= 0:
        #     return (index / T,)
    
    # if x > 0:
    #     return (1000,)

    final_state_error = abs(x - xtarget)
    fitness = T + PENALTY * final_state_error
    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def mutate(individual, mu, sigma, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            if i < N_CTRL_POINTS:
                individual[i] += random.gauss(mu, sigma)
                individual[i] = max(-1.0, min(1.0, individual[i]))
            else:
                individual[i] += random.gauss(mu, sigma * abs(x0) / 20.0)
                individual[i] = max(0.1, individual[i])
    return individual,

toolbox.register("mutate", mutate, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(40)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    # Evaluate the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the initial population's best individual
    hof.update(pop)
    
    NGEN = 0
    MAXGEN = 5000
    FITNESS_THRESHOLD = 6.0

    best_fitness_history = []

    while NGEN < MAXGEN:
        best_fitness = hof[0].fitness.values[0]
        if best_fitness <= 9.0:
            elites = tools.selBest(pop, 500)
        else:
            elites = tools.selBest(pop, 200)

        elites = list(map(toolbox.clone, elites))

        offspring = toolbox.select(pop, len(pop) - len(elites))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                
                # Clipping. Remove?
                for i in range(len(child1)):
                    child1[i] = max(-1.0, min(1.0, child1[i]))
                    child2[i] = max(-1.0, min(1.0, child2[i]))
                
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)
        pop[:] = offspring + elites

        NGEN += 1

        best_fitness = hof[0].fitness.values[0]
        best_fitness_history.append(best_fitness)

        print(f"Generation {NGEN}: Best fitness = {best_fitness:.4f}")

        if best_fitness < FITNESS_THRESHOLD:
            print(f"\nConvergence. Best fitness ({best_fitness:.4f}) is below threshold ({FITNESS_THRESHOLD}).")
            break

    best_ind = hof[0]
    best_u_sequence = np.array(best_ind[:-1])
    best_T = best_ind[-1]
    best_fitness = best_ind.fitness.values[0]

    print("\n--- Results ---")
    print(f"Optimal Time (T): {best_T:.4f}")
    print(f"Final fitness: {best_fitness:.4f}")
    print(f"Best individual: {best_ind}")

    # Simulate
    dt = best_T / N_CTRL_POINTS
    t_history = np.linspace(0, best_T, N_CTRL_POINTS + 1)
    x_history = [x0]
    x_val = x0
    for u in best_u_sequence:
        x_val += u * dt
        x_history.append(x_val)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.step(t_history[:-1], best_u_sequence, where='post', color='blue')
    plt.ylim(-1.1, 1.1)
    plt.title('Optimal Control Signal $u(t)$')
    plt.xlabel('Time (t)')
    plt.ylabel('Control Input $u$')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t_history, x_history, color='red')
    plt.plot(best_T, 0, 'go', label='Final State $x(T)=0$')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('State Trajectory $x(t)$')
    plt.xlabel('Time (t)')
    plt.ylabel('State $x$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bangbang_example.png')

    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Best Fitness', color='green')
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('bangbang_fitness_history.png')

if __name__ == "__main__":
    main()