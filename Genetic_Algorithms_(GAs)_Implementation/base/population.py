from base.individual import *

def generate_population(geo_matrix):

    def generate_pop(pop_size):
        """
            Generate a population of individuals.

            Parameters:
                - pop_size (int): The size of the population to generate.

            Returns:
                - list: A list containing the specified number of individuals. Each individual is represented as a list of areas.
        """
        # Defining a list with the areas of the geo_matrix
        areas = list(geo_matrix.keys())

        # Returning a population of individuals
        return [generate_valid_individual(areas) for individual_from_pop in range(pop_size)]

    return generate_pop


def evaluate_population_geo(geo_matrix):

    def pop_evaluation(population):
        """
            Evaluate the fitness of each individual in a population.

            Parameters:
                - population (list): A list of individuals, where each individual is represented as a list of areas.

        Returns:
            - list: A list of fitness scores, where each score corresponds to the fitness of the corresponding individual
              in the population.
        """
        # Evaluating each solution in the population
        return [evaluate_individual_geo(geo_matrix)(individual) for individual in population]

    return pop_evaluation
