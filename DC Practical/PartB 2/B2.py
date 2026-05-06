# Vaibhav Gaikwad BE A 46

import random
import numpy as np
from deap import base, creator, tools

# -----------------------------
# Fix randomness (same output every run)
# -----------------------------
random.seed(42)

# -----------------------------
# Define Fitness & Individual
# -----------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# -----------------------------
# Toolbox Setup
# -----------------------------
toolbox = base.Toolbox()

# Binary gene (0 or 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# Individual (list of 100 bits)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)

# Population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------
# Fitness Function
# -----------------------------
def evaluate(individual):
    """Maximize number of 1s"""
    return (sum(individual),)

toolbox.register("evaluate", evaluate)

# Genetic Operators
toolbox.register("mate", tools.cxTwoPoint)               # Crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Mutation
toolbox.register("select", tools.selTournament, tournsize=3) # Selection

# -----------------------------
# Main Function
# -----------------------------
def main():
    pop = toolbox.population(n=50)

    # Parameters
    CXPB = 0.5   # Crossover probability
    MUTPB = 0.2  # Mutation probability
    NGEN = 40    # Generations

    print("Starting Genetic Algorithm...\n")

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Evolution Loop
    for gen in range(NGEN):
        print(f"Generation {gen}")

        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Best fitness in this generation
        fits = [ind.fitness.values[0] for ind in pop]
        best_fit = max(fits)

        print(f" Best Fitness: {best_fit}/100\n")

        # Early stopping if optimal solution found
        if best_fit == 100:
            print("Optimal solution found early!\n")
            break

    # Final result
    print("---- End of Evolution ----")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best Individual Fitness:", best_ind.fitness.values[0])
    print("Best Individual (first 20 bits):", best_ind[:20])

# -----------------------------
# Run Program
# -----------------------------
if __name__ == "__main__":
    main()