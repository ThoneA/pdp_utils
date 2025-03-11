
import time
from pdp_utils import *

# files = ['Call_7_Vehicle_3.txt', 
#          'Call_300_Vehicle_90.txt',
#          'Call_130_Vehicle_40.txt',
#          'Call_80_Vehicle_20.txt',
#          'Call_35_Vehicle_7.txt',
#          'Call_18_Vehicle_5.txt'
#         ]

# files = ['Call_80_Vehicle_20.txt']

files = ['Call_7_Vehicle_3.txt']

num_runs = 10

results = {}

for file in files:
    prob = load_problem('pdp_utils/data/pd_problem/' + file)
    objective_values = []
    best_sol = None
    best_obj = float('inf')
    start_time = time.time()
    
    for _ in range(num_runs):
        rand_sol = random_function(prob)
        cost = cost_function(rand_sol, prob)
        objective_values.append(cost)
        
        if cost < best_obj:
            best_obj = cost
            best_sol = rand_sol
    
    end_time = time.time()
    running_time = end_time - start_time
    
    average_obj = sum(objective_values) / len(objective_values)
    best_obj = min(objective_values)
    improvement = 100 * (objective_values[0] - best_obj) / objective_values[0]
    feasiblity, c = feasibility_check(best_sol, prob)

    
    results[file] = {
        'Average Objective': average_obj, 
        'Best Objective': best_obj, 
        'Improvement %': improvement, 
        'Running time (s)': running_time,
        'Best sol': best_sol,
        'Feasibility': c
    }
    
    # initial_sol = initial_solution(prob)
    # local_search_sol = local_search(prob, initial_sol)
    
for file, stats in results.items():
    print(f"\nResults for {file}:")
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:2f}")
        else:
            print(f"{key}: {value}")











# prob = load_problem('pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt')

# sol = random_function(prob)

# print(prob.keys())

# feasiblity, c = feasibility_check(sol, prob)

# Cost = cost_function(sol, prob)

# print(feasiblity)
# print(c)
# print(Cost)