import time
from assignment_5 import *
from pdp_utils import *
from pdp_utils.operators import *
import traceback
import os
import gc

files = [
        'Call_7_Vehicle_3.txt', 
        # 'Call_18_Vehicle_5.txt',
        # 'Call_35_Vehicle_7.txt',
        # 'Call_80_Vehicle_20.txt',
        # 'Call_130_Vehicle_40.txt',
        # 'Call_300_Vehicle_90.txt',  
        ]

num_runs = 1
results = {}

# Clear memory before starting
gc.collect()

for file in files:
    print(f"\n===== Running for {file} =====")
    prob = load_problem('pdp_utils/data/pd_problem/' + file)
    objective_values = []
    best_sol = None
    best_obj = float('inf')
    start_time = time.time()
    
    # Generate initial solution once per file
    initial_sol = initial_solution(prob)
    all_histories = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Clear memory between runs
        gc.collect()
        
        try:
            # Run the optimized metaheuristic
            sol, op_stats = general_adaptive_metaheuristics(prob, initial_sol.copy())
            
            # Verify solution
            feasiblity, c = feasibility_check(sol, prob)
            if feasiblity:
                cost = cost_function(sol, prob)
                objective_values.append(cost)
                
                print(f"Run {run+1} completed with cost: {cost}")

                if cost < best_obj:
                    best_obj = cost
                    best_sol = sol.copy()
                    best_op_stats = op_stats.copy()
                    print(f"New best solution found: {best_obj}")
            else:
                print("Warning: Infeasible solution produced")
                
        except Exception as e:
            traceback.print_exc() 
            print(f"Error in run {run+1}: {e}")
    
    end_time = time.time()
    running_time = end_time - start_time
    average_run_time = running_time / num_runs if num_runs > 0 else 0

    print(f"\nCompleted all runs for {file}")
    print(f"Total running time: {running_time:.2f} seconds")
    
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

# Print overall results
print("\n===== OVERALL RESULTS =====")
for file, stats in results.items():
    print(f"\nResults for {file}:")
    for key, value in stats.items():
        if key == 'Best sol':
            print(f"{key}: {value}")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Print operator statistics
    if 'Best Objective' in stats:
        print("\nOperator Performance:")
        for op_name, stats in best_op_stats.items():
            print(f"{op_name}: Score={stats['score']}, Final Probability={stats['probability']:.4f}")