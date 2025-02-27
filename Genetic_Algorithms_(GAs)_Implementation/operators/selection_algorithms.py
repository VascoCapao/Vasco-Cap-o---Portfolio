import numpy as np

from base.individual import *

def tournament_selection_max(ts):
    """
        Perform tournament selection to select an individual with maximum fitness from a population.

        Parameters:
            - ts (int): The tournament size, i.e., the number of individuals to compete in each tournament.

        Returns:
            - function: A function that performs tournament selection on a population based on their fitness.
    """
    def inner_tournament(population, fitness):
        """
            Inner function that performs tournament selection on a population.

            Parameters:
                - population (list): A list of individuals.
                - fitness (dict): A dictionary mapping individuals to their fitness values.

            Returns:
                - list: The selected individual with maximum fitness.
        """
        # Randomly selecting ts number of individuals from the population
        # or, more specifically, choosing the individuals from the population via their index
        pool = random.choices([i for i in range(len(population))], k=ts)

        # Getting the fitness of the individuals of the given index
        pool_fits = [fitness[individual] for individual in pool]

        # Finding out where in the pool fits the best fitness is
        best = np.argmax(pool_fits)

        # Return the individual from the population whose index is the same as the index
        # in pool of the individual who was best in pool_fits
        return population[pool[best]]

    return inner_tournament

def fit_probability_selection_max():

    def inner_fit(population, fitness):
        """
            Inner function that performs fitness proportional selection on a population.

            Parameters:
                - population (list): A list of individuals.
                - fitness (dict): A dictionary mapping individuals to their fitness values.

            Returns:
                - list: The selected individual with maximum fitness.
        """
        # Getting the fitness of the individuals of the given index
        pool_fits = [fitness[individual] for individual in range(len(population))]

        # Calculating the probabilities of considering the fitness
        probabilities = [pool_fits[i]/sum(pool_fits) for i in range(len(pool_fits))]

        # Selecting the individual considering the probabilities computed
        selected_individual = random.choices([i for i in range(len(population))], weights=probabilities, k=1)

        # Returning the individual selected
        return population[selected_individual[0]]

    return inner_fit

def ranking_selection_max():

    def inner_ranking(population, fitness):
        """
            Inner function that performs ranking selection on a population.

            Parameters:
                - population (list): A list of individuals.
                - fitness (dict): A dictionary mapping individuals to their fitness values.

            Returns:
                - list: The selected individual with maximum fitness.
        """
        # Getting the fitness of the individuals of the given indexes
        pool_fits = [fitness[individual] for individual in range(len(population))]

        # Sort the fitness in ascending order
        pool_fits_sorted = sorted(pool_fits)

        # Calculate the probability via ranking
        sum_indices = sum(range(1, len(pool_fits_sorted) + 1))
        probabilities = [i / sum_indices for i in range(1, len(pool_fits_sorted) + 1)]

        # Selecting the individual based on the probabilities computed
        selected_individual = random.choices(range(len(population)), weights=probabilities, k=1)

        # Returning the individual selected
        return population[selected_individual[0]]

    return inner_ranking