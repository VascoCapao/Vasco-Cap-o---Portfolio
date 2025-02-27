from algorithm.algorithm import GA
from base.population import *
from operators.selection_algorithms import tournament_selection_max
from operators.crossovers import two_point_xover
from operators.mutators import displacement_mutation
from utils.utils import *


# Convert your list of lists into a dataframe, replace 'list_of_lists' and 'indexes' with your actual data and index names

list_of_lists = pass # Replace pass with your list of lists
indexes = pass # Replace pass with your index names

geo_matrix = list_to_dataframe(list_of_lists=list_of_lists, indexes=indexes)

# Parameters for the best configuration found
initializer = generate_population(geo_matrix)
evaluator = evaluate_population_geo(geo_matrix)
selector = tournament_selection_max(30)
xover = two_point_xover
mutator = displacement_mutation
elite_func = get_n_elite_max()

# Evolutionary algorithm parameters
pop_size = 100   
n_gens = 60  
p_xo = 0.85 
p_m = 0.15  

# Run the Genetic Algorithm (GA) with the specified parameters
GA(initializer=initializer,
   evaluator=evaluator,
   selector=selector,
   crossover=xover,
   mutator=mutator,
   pop_size=pop_size,
   n_gens=n_gens,
   p_xo=p_xo,
   p_m=p_m,
   verbose=True,  
   log_path="log/test_log.csv",
   elitism=True,  
   elite_func=elite_func,  
   seed=1)  


