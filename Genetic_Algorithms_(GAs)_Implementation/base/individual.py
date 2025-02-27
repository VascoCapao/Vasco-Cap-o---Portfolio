import random

def generate_valid_individual(areas):
    """
        Creating an individual in the population.

        Parameters:
            - areas (list): A list of areas that must represent the individual.

        Returns:
            - list: A valid individual after applying the specified rules.
    """
    # Shuffle the areas randomly
    random.shuffle(areas)

    # Remove 'D' if it is in the list
    individual = [element for element in areas if element != 'D']

    # Swap 'CS' and the following area if 'QG' is just before 'CS'
    if 'QG' in individual and 'CS' in individual:
        qg_index = individual.index('QG')
        cs_index = individual.index('CS')
        if qg_index + 1 == cs_index:
            if cs_index + 1 < len(individual):
                individual[cs_index], individual[cs_index + 1] = individual[cs_index + 1], individual[cs_index]
            else:
                individual[cs_index], individual[qg_index] = individual[qg_index], individual[cs_index]

    # Ensure 'RG' is in the second half of the individual
    midpoint = len(individual) // 2
    if 'RG' not in individual[midpoint:]:
        individual.remove('RG')
        individual.insert(random.randint(midpoint, len(individual) - 1), 'RG')

    # Skipping 'KS' if 'DV' is immediately after 'QS'
    if 'QS' in individual and 'DV' in individual and 'KS' in individual:
        qs_index = individual.index('QS')
        dv_index = individual.index('DV')
        ks_index = individual.index('KS')
        if dv_index == qs_index + 1:
            individual.remove('KS')
            individual.insert(ks_index, '-')

    # Returning the individual
    return individual

def is_individual_valid(individual):
    """
        Check if an individual meets validity criteria.

        Parameters:
            - individual (list): The individual to be validated, represented as a list of areas.

        Returns:
            - bool: True if the individual meets all validity criteria, False otherwise.
    """
    # Areas that must be included in the individual
    areas = ['D', 'FC', 'G', 'QS', 'QG', 'CS', 'RG', 'DV', 'SN']

    # Checking if each area is included
    for area in areas:
        if area not in individual:
            return False

    # Defining the index of specific areas
    qs_index = individual.index('QS')
    dv_index = individual.index('DV')
    qg_index = individual.index('QG')
    cs_index = individual.index('CS')
    midpoint_index = len(individual) // 2

    # Checking if the individual fulfills the rules
    if (individual[0] != 'D' or individual[-1] != 'D') or ('-' in individual and dv_index != qs_index + 1) or \
            (cs_index == qg_index + 1) or ('RG' not in individual[midpoint_index:]):
        return False

    else:
        return True

def evaluate_individual_geo(geo_matrix):

    def get_fitness(individual):
        """
            Calculate the fitness of an individual based on geo.

            Parameters:
                - individual (list): A list of areas representing the individual's path.

            Returns:
                - float: The total geo gained if the individual is valid, otherwise -99999999.
        """
        # Add 'D' at the beginning and end of the individual
        individual_with_d = ['D'] + [str(area) for area in individual] + ['D']

        # Initiate the counter for the Geo gained
        total_geo = 0

        # Calculate total Geo gained
        for i in range(len(individual_with_d) - 1):
            from_area = individual_with_d[i]
            to_area = individual_with_d[i + 1]

            # If the origin or destination area is not in the matrix, assign 0 to the Geo value
            if from_area not in geo_matrix.index or to_area not in geo_matrix.columns:
                total_geo += 0
            else:
                total_geo += geo_matrix.loc[from_area, to_area]

        # If the individual is valid, return the total_geo, otherwise, -99999999
        return total_geo if is_individual_valid(individual_with_d) else -99999999

    return get_fitness