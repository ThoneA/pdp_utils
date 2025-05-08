import time
from assignment_5 import *
from pdp_utils import *
import traceback
import os
import gc

# WARNING!!!:
# If you run this script, it will overwrite the log file in the log folder.
# Make sure to back up any important data before running this script.

files = [
        # 'Call_7_Vehicle_3.txt', 
        # 'Call_18_Vehicle_5.txt',
        # 'Call_35_Vehicle_7.txt',
        # 'Call_80_Vehicle_20.txt',
        'Call_130_Vehicle_40.txt',
        # 'Call_300_Vehicle_90.txt',  
        ]

num_runs = 1
results = {}


for file in files:
    # Clear memory before starting
    gc.collect()    
    # Clear the results log file at the start of the script
    print(f"\n===== Running for {file} =====")
    prob = load_problem('pdp_utils/data/pd_problem/' + file)
    objective_values = []
    best_sol = None
    best_obj = float('inf')
    start_time = time.time()
    # all_histories = []
    # Clear the results log file at the start of the script
    with open(f"log/{str(prob['n_calls'])}_log.txt", "w") as log_file:
        log_file.write(f"Starting call_{prob['n_calls']}\n")
    
    with open(f"log/new_best/{str(prob['n_calls'])}_new_best_log.txt", "w") as n_w_file:
        n_w_file.write(f"Starting call_{prob['n_calls']}\n")
    
    # Generate initial solution once per file
    initial_sol = initial_solution(prob)
    # all_histories = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Clear memory between runs
        gc.collect()
        
        try:
            # Run the optimized metaheuristic
            sol, op_stats, history = general_adaptive_metaheuristics(prob, initial_sol.copy())
            
            # Verify solution
            feasiblity, c = feasibility_check(sol, prob)
            if feasiblity:
                cost = cost_function(sol, prob)
                objective_values.append(cost)
                # all_histories.append(history)
                
                print(f"Run {run+1} completed with cost: {cost}")

                if cost < best_obj:
                    best_obj = cost
                    best_sol = sol.copy()
                    best_op_stats = op_stats.copy()
                    # best_history = history
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
        with open(f"log/{str(prob['n_calls'])}_log.txt", "a") as log_file:
            log_file.write(f"\n \nResults for {file}:\n")
            log_file.write(f"Best Objective: {best_obj:.2f}\n")
            log_file.write(f"Improvement %: {improvement:.2f}\n")
            log_file.write(f"Running time (s): {running_time:.2f}\n")
            log_file.write(f"Feasibility: {c}\n")
            log_file.write(f"Best sol: {best_sol}\n")
        
        
        # Print results for the current file
        print(f"\nResults for {file}:")
        print(f"Average Objective: {average_obj:.2f}")
        print(f"Best Objective: {best_obj:.2f}")
        print(f"Improvement %: {improvement:.2f}")
        print(f"Running time (s): {running_time:.2f}")
        print(f"Average Running time (s): {average_run_time:.2f}")
        print(f"Feasibility: {c}")
        print(f"Best sol: {best_sol}")
        
    else:
        print("No feasible solution found")
        results[file] = {
            'Status': 'No feasible solution found',
            'Running time (s)': running_time
        }
        print(f"\nResults for {file}:")
        print(f"Status: No feasible solution found")
        print(f"Running time (s): {running_time:.2f}")
            
    # if best_sol is not None:
    #     plot_optimization_history(best_history, file, save_fig=True)
    
    # if all_histories:
    #     plot_convergence_statistics(all_histories, file, save_fig=True)

# Print overall results
# print("\n===== OVERALL RESULTS =====")
# for file, stats in results.items():
#     print(f"\nResults for {file}:")
#     for key, value in stats.items():
#         if key == 'Best sol':
#             print(f"{key}: {value}")
#         elif isinstance(value, (int, float)):
#             print(f"{key}: {value:.2f}")
#         else:
#             print(f"{key}: {value}")
    
#     # Print operator statistics
#     if 'Best Objective' in stats:
#         print("\nOperator Performance:")
#         for op_name, stats in best_op_stats.items():
#             print(f"{op_name}: Score={stats['score']}, Final Probability={stats['probability']:.4f}")