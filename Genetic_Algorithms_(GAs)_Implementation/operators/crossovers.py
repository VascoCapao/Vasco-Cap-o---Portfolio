import random

import random

def one_point_xover(p1, p2):
    """
        Perform one-point crossover between two parent individuals to generate offspring.
        In one-point crossover, a random crossover point is chosen, and elements before this point are inherited from
        one parent while elements after this point are inherited from the other parent.

        Parameters:
            - p1 (list): The first parent individual.
            - p2 (list): The second parent individual.

        Returns:
            - o1 (list): The first offspring individual generated from the crossover operation.
            - o2 (list): The second offspring individual generated from the crossover operation.
    """
    
    # Choosing a crossover point
    xover_point = random.randint(1, len(p1)-1)

    # Generating the offspring
    o1 = p1[:xover_point] + p2[xover_point:]
    o2 = p2[:xover_point] + p1[xover_point:]

    # Returning the offspring
    return o1, o2

def two_point_xover(p1, p2):
    """
        Perform two-point crossover between two parent individuals to generate offspring.
        In two-point crossover, two random crossover points are chosen, and elements between these points
        are inherited from one parent while elements outside these points are inherited from the other parent.

        Parameters:
            - p1 (list): The first parent individual.
            - p2 (list): The second parent individual.

        Returns:
            - o1 (list): The first offspring individual generated from the crossover operation.
            - o2 (list): The second offspring individual generated from the crossover operation.
    """
    
    # Choosing the random xover points
    xover_point1 = random.randint(1, len(p1) - 2)
    xover_point2 = random.randint(xover_point1 + 1, len(p1) - 1)

    # Generating the offspring
    o1 = p1[:xover_point1] + p2[xover_point1:xover_point2] + p1[xover_point2:]
    o2 = p2[:xover_point1] + p1[xover_point1:xover_point2] + p2[xover_point2:]

    # Returning the offspring
    return o1, o2


def pmx_crossover(p1, p2):
    """
        Perform partially-mapped crossover between two parent individuals to generate offspring.
        It involves selecting a random crossover range, then creating offspring by exchanging the selected range between
        parents, followed by validating the offspring using mapping relationships.

        Parameters:
            - p1 (list): The first parent individual.
            - p2 (list): The second parent individual.

        Returns:
            - o1 (list): The first offspring individual generated from the crossover operation.
            - o2 (list): The second offspring individual generated from the crossover operation.
    """
    size = len(p1)

    # Select crossover range at random
    xover_point1, xover_point2 = sorted(random.sample(range(1, size - 2), 2))

    # Create offspring by exchanging the selected range
    o1 = p1[:xover_point1] + p2[xover_point1:xover_point2] + p1[xover_point2:]
    o2 = p2[:xover_point1] + p1[xover_point1:xover_point2] + p2[xover_point2:]

    # Determine the mapping relationship to validate offspring
    mapping1 = {p2[i]: p1[i] for i in range(xover_point1, xover_point2)}
    mapping2 = {p1[i]: p2[i] for i in range(xover_point1, xover_point2)}

    # Validate children with the mapping relationship
    for i in list(range(xover_point1)) + list(range(xover_point2, size)):
        if o1[i] in mapping1:
            while o1[i] in mapping1:
                o1[i] = mapping1[o1[i]]
        if o2[i] in mapping2:
            while o2[i] in mapping2:
                o2[i] = mapping2[o2[i]]

    # Returning the offspring
    return o1, o2

def ox_crossover(p1, p2):
    """
        Perform Order Crossover (OX) between two parent individuals to generate offspring.

        OX involves selecting a random crossover range, then creating offspring by copying the selected range from both
        parents to offspring. After copying, the remaining genes in each offspring are filled by using the remaining
        genes from the other parent.

        Parameters:
            - p1 (list): The first parent individual.
            - p2 (list): The second parent individual.

        Returns:
            - o1 (list): The first offspring individual generated from the crossover operation.
            - o2 (list): The second offspring individual generated from the crossover operation.
    """
    size = len(p1)

    # Select crossover range at random
    xover_point1, xover_point2 = sorted(random.sample(range(size), 2))
    
    # Create empty offspring
    o1 = [-1] * size
    o2 = [-1] * size
    
    # Copy the selected range from p1 and p2 to offspring, respectively
    o1[xover_point1:xover_point2] = p1[xover_point1:xover_point2]
    o2[xover_point1:xover_point2] = p2[xover_point1:xover_point2]
    
    # Fill the remaining genes in o1 using p2 and vice versa
    idx1 = xover_point2
    idx2 = xover_point2
    for i in range(size):
        if idx1 == size:
            idx1 = 0
        if idx2 == size:
            idx2 = 0
        if p2[i] not in o1:
            o1[idx1] = p2[i]
            idx1 += 1
        if p1[i] not in o2:
            o2[idx2] = p1[i]
            idx2 += 1

    # Returning the offspring
    return o1, o2


