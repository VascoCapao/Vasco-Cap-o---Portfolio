import pandas as pd

# Specify the column names
column_names = ['selector', 'crossover', 'mutator', 'p_xo', 'p_m', 'pop_s', 'gen', 'seed', 'best_individual', 'best_fit']

# Load the CSV file using the relative path and specifying the column names
results1 = pd.read_csv("log\gridsearch_log.csv", names=column_names)
results2 = pd.read_csv("log\gridsearch_log2.csv", names=column_names)

grouped1 = results1.groupby(['selector', 'crossover', 'mutator', 'p_xo', 'p_m', 'pop_s', 'gen',]).agg({'best_fit': ['mean']})
grouped1.columns = ['fitness_mean']
grouped1 = grouped1.reset_index()
best_parameters1 = grouped1.sort_values(by='fitness_mean', ascending=False).head()

grouped2 = results2.groupby(['selector', 'crossover', 'mutator', 'p_xo', 'p_m', 'pop_s', 'gen',]).agg({'best_fit': ['mean']})
grouped2.columns = ['fitness_mean']
grouped2 = grouped2.reset_index()
best_parameters2 = grouped2.sort_values(by='fitness_mean', ascending=False).head()

print(best_parameters1)
print(100*'-')
print(best_parameters2)


'''
RESULTS:

1º Parameters for GridSearch

              selector      crossover                mutator  p_xo  p_m  pop_s  gen  fitness_mean
3077  inner_tournament  pmx_crossover  displacement_mutation   0.9  0.3     50   30   7596.842667
3047  inner_tournament  pmx_crossover  displacement_mutation   0.8  0.3     40   30   7594.466667
3076  inner_tournament  pmx_crossover  displacement_mutation   0.9  0.3     50   20   7584.242667
2564  inner_tournament   ox_crossover  displacement_mutation   0.8  0.3     50   30   7583.066667
2534  inner_tournament   ox_crossover  displacement_mutation   0.7  0.3     40   30   7573.533333
----------------------------------------------------------------------------------------------------

2º Parameters for GridSearch
              selector        crossover                mutator  p_xo   p_m  pop_s  gen  fitness_mean
5471  inner_tournament  two_point_xover  displacement_mutation  0.85  0.15    100   60   7758.066667 -> Best parameters
4046  inner_tournament  one_point_xover  displacement_mutation  0.95  0.25     80   60   7743.200000
4044  inner_tournament  one_point_xover  displacement_mutation  0.95  0.25     80   40   7743.200000
4045  inner_tournament  one_point_xover  displacement_mutation  0.95  0.25     80   50   7743.200000
5498  inner_tournament  two_point_xover  displacement_mutation  0.95  0.15    100   60   7736.933333
'''