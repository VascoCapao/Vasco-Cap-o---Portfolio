import csv
from copy import deepcopy
import random
import numpy as np

def GA(initializer, evaluator, selector, crossover, mutator, pop_size, n_gens, p_xo, p_m, elite_func,
       verbose=False, log_path=None, elitism=True, seed=0):
    """
        Genetic Algorithm (GA) implementation.

        Parameters:
            - initializer (function): A function to initialize the population.
            - evaluator (function): A function to evaluate the fitness of individuals in the population.
            - selector (function): A function to select parents from the population.
            - crossover (function): A function to perform crossover on two parents.
            - mutator (function): A function to mutate individuals.
            - pop_size (int): Population size.
            - n_gens (int): Number of generations.
            - p_xo (float): Probability of crossover.
            - p_m (float): Probability of mutation.
            - elite_func (function): A function to select elite individuals.
            - verbose (bool, optional): Indicates if there should be information printed (True), or not (False), on each
                                   generation and final solution.
            - log_path (str, optional): Indicates if the values at each generation should be stored in a csv file.
            - elitism (bool, optional): Boolean that indicates if there should be elitism in the algorithm.
            - seed (int, optional): Random seed. Defaults to 0.

        Returns:
            - list: the best individual in the population.
            - integer: fitness score of that individual.
    """
    # Setting up the seed
    random.seed(seed)
    np.random.seed(seed)

    if elite_func is None:
        raise Exception("Without a proper elite function I will not work.")

    # Initializing the gen 0 population
    population = initializer(pop_size)

    # Evaluating the current population
    pop_fit = evaluator(population)

    for gen in range(n_gens):

        # Creating an empty offspring population
        offspring = []

        # While the offspring population is not full:
        while len(offspring) < pop_size:

            # Selecting the parents
            p1, p2 = selector(population, pop_fit), selector(population, pop_fit)

            # Choosing between crossover and reproduction
            if random.random() < p_xo:
                # Do crossover
                o1, o2 = crossover(p1, p2)

            else:
                # Reproduction
                o1, o2 = deepcopy(p1), deepcopy(p2)

            # Mutating the offspring
            o1, o2 = mutator(o1, p_m), mutator(o2, p_m)

            # Adding the offspring into the offspring population
            offspring.extend([o1, o2])

        # Making sure offspring population does not exceed pop_size
        while len(offspring) > pop_size:
            offspring.pop()

        # Make sure the elite of the offspring population is inserted into the offspring into the next generation
        if elitism:
            elite, best_fit = elite_func(population, pop_fit)
            # Adding the elite, unchanged into the offspring population
            offspring[-1] = elite

        # Replacing the current population with the offspring population
        population = offspring

        # Evaluating the current population
        pop_fit = evaluator(population)

        # Displaying and logging the generation results
        new_elite, new_fit = elite_func(population, pop_fit)

        if verbose:
            print(f'        {gen}        |        {new_fit}        ')
            print('-' * 32)

        if log_path is not None:
            with open(log_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([seed, gen, new_fit, new_elite])

    # Inserting 'Dirtmouth' into the beginning and ending of each the individual
    for individual in population:
        individual.insert(0, 'D')
        individual.append('D')

    # Displaying the individual with the best fitness
    if verbose:
        print('Final solution:', population[pop_fit.index(max(pop_fit))])
        print('Best fitness:', max(pop_fit))

    # Returning the best individual and its fitness
    return population[pop_fit.index(max(pop_fit))], max(pop_fit)

