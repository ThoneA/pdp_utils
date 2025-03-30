import time
from pdp_utils import *
from pdp_utils.operators import *
import traceback

# files = [ 
#           'Call_80_Vehicle_20.txt',
#           'Call_35_Vehicle_7.txt'
#          ]

files = [
        # 'Call_7_Vehicle_3.txt', 
        #   'Call_300_Vehicle_90.txt',
          'Call_130_Vehicle_40.txt',
        #   'Call_80_Vehicle_20.txt',
        #   'Call_35_Vehicle_7.txt',
        #   'Call_18_Vehicle_5.txt'
        ]
# files = ['Call_80_Vehicle_20.txt']

# files = ['Call_7_Vehicle_3.txt']

num_runs = 10
results = {}


for file in files:
    print(f"Running for {file}")
    prob = load_problem('pdp_utils/data/pd_problem/' + file)
    objective_values = []
    best_sol = None
    best_obj = float('inf')
    start_time = time.time()
    initial_sol = initial_solution(prob)
    
    for run in range(num_runs):
        print(f"Run {run}")
        try:
            sol = upgraded_simulated_annealing(prob, initial_sol)
            cost = cost_function(sol, prob)
            feasiblity, c = feasibility_check(sol, prob)
            
            if feasiblity:
                objective_values.append(cost)

                if cost < best_obj:
                    best_obj = cost
                    best_sol = sol
            else:
                print("Infeasible solution")
        except Exception as e:
            # traceback.print_exc() 
            print(f"Error: {e}")
    
        
    end_time = time.time()
    running_time = end_time - start_time
    average_run_time = running_time / num_runs
    
    if objective_values:
        average_obj = sum(objective_values) / len(objective_values)
        best_obj = min(objective_values)
        init_cost = cost_function(initial_sol, prob)
        improvement = 100 * (init_cost - best_obj) / init_cost
        feasiblity, c = feasibility_check(best_sol, prob)
    
        results[file] = {
            'Average Objective': average_obj, 
            'Best Objective': best_obj, 
            'Improvement %': improvement, 
            'Running time (s)': running_time,
            'Average Running time (s)': average_run_time,
            'Feasibility': c,
            'Best sol': best_sol
        }
        
    else:
        print("No feasible solution found")
        results[file] = {
            'Status': 'No feasible solution found',
            'Running time (s)': running_time
        }
    
for file, stats in results.items():
    print(f"\nResults for {file}:")
    for key, value in stats.items():
        if key == 'Best sol':
            print(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
            
