import numpy as np
import pandas as pd
import random

def get_n_elite_max():

    def get_elite(population, pop_fit):
        """
            Inner function that retrieves the top elite individual from a population along with its fitness value.

            Parameters:
                - population (list): A list of individuals.
                - pop_fit (list): A list of fitness values corresponding to the individuals in the population.

            Returns:
                - individual: The top elite individual from the population.
                - float: The fitness value corresponding to the top elite individual.
        """
        # Getting the best n elites (only the top one if n=1)
        best_index = np.argmax(pop_fit)

        # Returning the elite individual and its fitness
        return population[best_index], pop_fit[best_index]

    return get_elite


def list_to_dataframe(list_of_lists, indexes):
    """
        Convert a list of lists into a pandas DataFrame with specified indexes.

        Parameters:
            - list_of_lists (list): A list of lists representing the data.
            - indexes (list): A list of index labels to be used for rows and columns in the DataFrame.

        Returns:
            - pandas.DataFrame: A DataFrame with the data from the list_of_lists, where rows and columns are
              labelled using the provided indexes.
    """
    df = pd.DataFrame(list_of_lists, columns=indexes, index=indexes)
    return df


def generate_geo_dataframe(geo_names, negative_prob=0.2,seed=0):
    """
        Generates a non-symmetric Geo DataFrame with the following rules:

        - Diagonal is 0.
        - More positive than negative values.
        - G to FC gain is at least 3.2% less than the minimum positive gain (excluding diagonal).

        Parameters:
            - geo_names: List of Geo location names.
            - negative_prob: Probability of a value being negative (default 0.2).

        Returns:
            - pandas.DataFrame: A DataFrame representing the Geo values.
    """
    # Setting up the seed
    random.seed(seed)
    np.random.seed(seed)

    num_geos = len(geo_names)
    geo_matrix = np.zeros((num_geos, num_geos))

    # Generate random gains/losses (excluding diagonal)
    for i in range(num_geos):
        for j in range(num_geos):
            if i != j: 
                if random.random() < negative_prob:
                    geo_matrix[i, j] = -random.randint(10, 1000)
                else:
                    geo_matrix[i, j] = random.randint(10, 1000)

    # Find positive gains (excluding G to FC and diagonal)
    g_index = geo_names.index("G")
    fc_index = geo_names.index("FC")
    positive_gains = geo_matrix[
        # Exclude G row
        (geo_matrix > 0) & (np.arange(num_geos) != g_index)[:, None]
        # Exclude FC column
        & (np.arange(num_geos) != fc_index)
        # Exclude diagonal
        & (np.arange(num_geos) != np.arange(num_geos)[:, None])
    ]
    min_positive_gain = positive_gains.min()

    # Adjust G to FC gain
    geo_matrix[g_index, fc_index] = max(0, min_positive_gain * 0.968)

    # Create DataFrame
    df = list_to_dataframe(geo_matrix, geo_names)

    # Return the DataFrame
    return df







