from copy import deepcopy
from base.individual import *

def swap_mutation(individual, p_m):
    """
       Perform a swap mutation on an individual with a given probability. Two distinct indices are selected randomly,
       and the elements at these indices are swapped.

       Parameters:
           - individual (list): The individual to be mutated, represented as a list of elements.
           - p_m (float): The probability of mutation occurring.

       Returns:
           - list: The mutated individual. If mutation does not occur, the original individual is returned.
    """
    # Create a copy of the individual
    mutated_individual = deepcopy(individual)

    # Check if mutation should occur based on probability p_m
    if random.random() < p_m:
        # Select two distinct indices randomly
        index1, index2 = random.sample(range(len(individual)), 2)

        # Swap the values at the selected indices
        mutated_individual[index1], mutated_individual[index2] = mutated_individual[index2], mutated_individual[index1]

    # Return the mutated individual
    return mutated_individual


def scramble_mutation(individual, p_m):
    """
        Perform a scramble mutation on an individual with a given probability.
        A subset of the individual is selected randomly and the  elements within this subset are shuffled.

        Parameters:
            - individual (list): The individual to be mutated.
            - p_m (float): The probability that a mutation will occur.

        Returns:
            - list: The mutated individual. If mutation does not occur, the original individual is returned.
    """
    # Create a copy of the individual
    mutated_individual = deepcopy(individual)

    # Check if mutation should occur based on probability p_m
    if random.random() < p_m:
        # Select two distinct indices randomly
        index1, index2 = random.sample(range(len(individual)), 2)

        # Ensure index1 is smaller than index2
        if index1 > index2:
            index1, index2 = index2, index1

        # Shuffle the subset of genes between index1 and index2
        subset = mutated_individual[index1:index2 + 1]
        random.shuffle(subset)
        mutated_individual[index1:index2 + 1] = subset

    # Return the mutated individual
    return mutated_individual


def inversion_mutation(individual, p_m):
    """
       Perform an inversion mutation on an individual with a given probability.
       A subset of the individual is selected randomly and the elements within this subset are reversed.

       Parameters:
           - individual (list): The individual to be mutated.
           - p_m (float): The probability that a mutation will occur.

       Returns:
           - list: The mutated individual. If mutation does not occur, the original individual is returned.
    """
    # Create a copy of the individual
    mutated_individual = deepcopy(individual)

    # Check if mutation should occur based on probability p_m
    if random.random() < p_m:
        # Select two distinct indices randomly
        index1, index2 = random.sample(range(len(individual)), 2)

        # Make sure index1 is smaller than index2
        if index1 > index2:
            index1, index2 = index2, index1

        # Reverse the subset of genes between index1 and index2
        subset = mutated_individual[index1:index2 + 1]
        subset.reverse()
        mutated_individual[index1:index2 + 1] = subset

    # Return the mutated individual
    return mutated_individual


def displacement_mutation(individual, p_m):
    """
        Perform a displacement mutation on an individual with a given probability.
        A random substring of the individual is selected, removed from its original position, and inserted into a new
        random position within the individual.

        Parameters:
            - individual (list): The individual to be mutated.
            - p_m (float): The probability that a mutation will occur.

        Returns:
            - list: The mutated individual. If mutation does not occur, the original individual is returned.
    """
    # Create a copy of the individual
    mutated_individual = deepcopy(individual)

    # Check if mutation should occur based on probability p_m
    if random.random() < p_m:
        # Select a random substring to displace
        start_index = random.randint(0, len(mutated_individual) - 1)
        end_index = random.randint(start_index + 1, len(mutated_individual))
        substring = mutated_individual[start_index:end_index]

        # Remove the selected substring from its original position
        del mutated_individual[start_index:end_index]

        # Choose a random position to insert the substring
        insert_index = random.randint(0, len(mutated_individual))

        # Insert the substring into the new position
        mutated_individual[insert_index:insert_index] = substring

    # Return the mutated individual
    return mutated_individual


def distance_based_mutation(individual, p_m):
    """
        Perform a distance-based mutation on an individual with a given probability.
        A random position is selected, and a random distance is used to define an interval around this position.
        The elements within this interval are shuffled.

        Parameters:
            - individual (list): The individual to be mutated.
            - p_m (float): The probability that a mutation will occur.

        Returns:
            - list: The mutated individual. If mutation does not occur, the original individual is returned.
    """
    # Create a copy of the individual
    mutated_individual = deepcopy(individual)

    # Check if mutation should occur based on probability p_m
    if random.random() < p_m:
        # Select a random position
        g = random.randint(0, len(mutated_individual) - 1)

        # Generate a random distance between 1 and 10
        distance = random.randint(1, 10)

        # Determine the interval around the selected position
        start_index = max(0, g - distance)
        end_index = min(len(mutated_individual), g + distance + 1)

        # Select the elements within the interval
        selected_elements = mutated_individual[start_index:end_index]

        # Shuffle the selected interval
        random.shuffle(selected_elements)

        # Update the chromosome with the shuffled interval
        mutated_individual[start_index:end_index] = selected_elements

    # Return the mutated individual
    return mutated_individual


def center_inversion_mutation(individual, p_m):
    """
       Perform a center inversion mutation on an individual with a given probability.
       The individual is divided into two parts at a random split point, and each part is reversed.

       Parameters:
           - individual (list): The individual to be mutated.
           - p_m (float): The probability that a mutation will occur.

       Returns:
           - list: The mutated individual. If mutation does not occur, the original individual is returned.
    """
    # Create a copy of the individual
    mutated_individual = deepcopy(individual)

    # Check if mutation should occur based on probability p_m
    if random.random() < p_m:
        # Determine the point to divide the individual into two parts randomly
        split_point = random.randint(1, len(mutated_individual) - 1)

        # Divide the individual into two parts
        first_part = mutated_individual[:split_point]
        second_part = mutated_individual[split_point:]

        # Reverse each part
        first_part = first_part[::-1]
        second_part = second_part[::-1]

        # Transfer the result to the offspring
        mutated_individual = first_part + second_part

    # Return the mutated individual
    return mutated_individual