import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from deap import base, creator, tools, algorithms

x10 = 5.0
x20 = 0.0
x1f = 0.0

N_CTRL_POINTS = 1000

PENALTY = 100.0

# GA parameters
POPULATION_SIZE = 1000
CXPB = 0.7  # Crossover probability
MUTPB = 0.2 # Mutation probability

T_FIXED = 10.0

# DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def gen_individual():
    u_values = [random.uniform(-1.0, 1.0) for _ in range(N_CTRL_POINTS)]
    return creator.Individual(u_values)

toolbox.register("individual", gen_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    u_sequence = individual

    dt = T_FIXED / N_CTRL_POINTS
    x1 = x10
    x2 = x20

    fitness = 0.0

    for u in u_sequence:
        fitness += (0.5 * u**2) * dt
        x1 += x2 * dt
        x2 += u * dt
    
    penalty = PENALTY * (x1 - x1f)**2
    fitness += penalty

    print(penalty)
    
    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

def mutate(individual, mu, sigma, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigma)
            # Clipping. Remove?
            individual[i] = max(-1.0, min(1.0, individual[i]))
    return individual,

toolbox.register("mutate", mutate, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    NGEN = 0
    MAXGEN = 5000
    
    last_best_fitness = float('inf')
    gens_wo_improvement = 0
    STAGNATION_LIMIT = 10

    while NGEN < MAXGEN:
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
        print(f"Generation {NGEN}: Best fitness = {best_fitness:.4f}")

        if best_fitness < last_best_fitness:
            last_best_fitness = best_fitness
            gens_wo_improvement = 0
        else:
            gens_wo_improvement += 1
        
        if gens_wo_improvement >= STAGNATION_LIMIT:
            print(f"\nConvergence. No improvement for {STAGNATION_LIMIT} generations.")
            break

    best_ind = hof[0]
    best_u_sequence = np.array(best_ind)
    best_fitness = best_ind.fitness.values[0]

    print("\n--- Results ---")
    print(f"Final fitness: {best_fitness:.4f}")
    print(f"Best individual: {best_ind}")

    # Simulate
    dt = T_FIXED / N_CTRL_POINTS
    t_history = np.linspace(0, T_FIXED, N_CTRL_POINTS + 1)
    x1_history = [x10]
    x2_history = [x20]
    x1_val = x10
    x2_val = x20
    for u in best_u_sequence:
        x1_val += x2_val * dt
        x2_val += u * dt
        x1_history.append(x1_val)
        x2_history.append(x2_val)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.step(t_history[:-1], best_u_sequence, where='post', color='blue')
    plt.title('Optimal Control Signal $u(t)$')
    plt.xlabel('Time (t)')
    plt.ylabel('Control Input $u$')
    plt.grid(True)
    plt.ylim(-1.1, 1.1)

    plt.subplot(3, 1, 2)
    plt.plot(t_history, x1_history, color='red')
    plt.plot(T_FIXED, x1f, 'go', label=f'Final State $x_1(T)={x1f}$')
    plt.axhline(x1f, color='gray', linestyle='--')
    plt.title('State Trajectory $x_1(t)$')
    plt.xlabel('Time (t)')
    plt.ylabel('State $x_1$')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t_history, x2_history, color='purple')
    plt.title('State Trajectory $x_2(t)$')
    plt.xlabel('Time (t)')
    plt.ylabel('State $x_2$')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('energy_example.png')

if __name__ == "__main__":
    main()