from algorithm.algorithm import GA
from base.population import *
from operators.selection_algorithms import *
from operators.crossovers import *
from operators.mutators import *
from utils.utils import *
from tqdm import tqdm
import itertools
import csv

# Parameters for the GA
pop_size = [60, 80, 100 ]
n_gens = [40, 50, 60]
elite_func = get_n_elite_max()
seeds = list(range(1, 16))  

# Lists of operators
selectors = [tournament_selection_max(30), fit_probability_selection_max(), ranking_selection_max()]
crossovers = [one_point_xover, two_point_xover, pmx_crossover, ox_crossover]
mutators = [swap_mutation, scramble_mutation, inversion_mutation, displacement_mutation, distance_based_mutation, center_inversion_mutation]
# Crossover probabilities
p_xo_range = [0.75, 0.85, 0.95]
# Mutation probabilities
p_m_range = [0.25, 0.15, 0.05]

# Initialize progress bar
total_iterations = len(selectors) * len(crossovers) * len(mutators) * len(p_xo_range) * len(p_m_range) * len(pop_size) * len(n_gens) * len(seeds)
progress_bar = tqdm(total=total_iterations)

# Iterate over all parameter combinations
for selector, crossover, mutator, p_xo, p_m, pop_s, gen, seed in itertools.product(
    selectors, crossovers, mutators, p_xo_range, p_m_range, pop_size, n_gens, seeds
):
    # Generate data for each run
    geo_matrix = generate_geo_dataframe(['D', 'FC', 'G', 'QS', 'QG', 'CS', 'KS', 'RG', 'DV', 'SN'], seed=seed)
    
    # Run GA
    best_individual, best_fit = GA(
        initializer=generate_population(geo_matrix),
        evaluator=evaluate_population_geo(geo_matrix),
        selector=selector,
        crossover=crossover,
        mutator=mutator,
        pop_size=pop_s,
        n_gens=gen,
        p_xo=p_xo,
        p_m=p_m,
        elite_func=elite_func,
        verbose=True,
        log_path="../log/test_log.csv",
        elitism=True,
        seed=seed,
    )

    # Log results
    with open("../log/gridsearch_log.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([selector.__name__, crossover.__name__, mutator.__name__, p_xo, p_m, pop_s, gen, seed, best_individual, best_fit])

    # Update progress bar
    progress_bar.update(1)    

progress_bar.close()


