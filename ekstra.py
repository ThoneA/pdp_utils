

# Example usage:
# best_solution, stats, history = general_adaptive_metaheuristics(prob, initial_sol)

        

# def general_adaptive_metaheuristics(prob, initial_sol, segment_size=100):
#     best_sol = initial_sol.copy()
#     incumbent = initial_sol.copy()
#     T_f = 0.1  
    
#     operators = [
#         {"name": "P1", "function": OP1},
#         {"name": "P2", "function": OP2},
#         {"name": "P3", "function": OP3},
#         {"name": "P4", "function": OP4},
#         {"name": "P5", "function": OP5},
#         {"name": "P6", "function": OP6},
#         {"name": "P7", "function": OP7},
#         {"name": "P8", "function": OP8},
#         {"name": "P9", "function": OP9}
#     ]

#     op_stats = {}
#     for op in operators:
#         op_stats[op["name"]] = {
#             "score": 0,
#             "counter": 0,
#             "probability": 1.0 / len(operators)
#         }
    
#     iterations_since_best = 0
#     escape_condition = 1000  # After this many iterations without improvement, apply escape

#     incumbent_cost = cost_function(incumbent, prob)
#     best_cost = incumbent_cost

#     delta_w = []
    
#     # Initial warmup phase
#     for w in range(1, 100): 
#         # Select operator based on current probabilities
#         probabilities = [op_stats[op["name"]]["probability"] for op in operators]
#         chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
#         chosen_op = operators[chosen_op_idx]
        
#         # Apply selected operator
#         op_stats[chosen_op["name"]]["counter"] += 1
#         new_sol = chosen_op["function"](prob, incumbent)
            
#         feasibility, _ = feasibility_check(new_sol, prob)
#         if not feasibility:
#             continue
        
#         new_cost = cost_function(new_sol, prob)
#         delta_E = new_cost - incumbent_cost
        
#         if feasibility and delta_E < 0:
#             incumbent = new_sol.copy()
#             incumbent_cost = new_cost
            
#             # Better than incumbent: +2 points
#             op_stats[chosen_op["name"]]["score"] += 2
            
#             if incumbent_cost < best_cost:
#                 best_sol = incumbent.copy()
#                 best_cost = incumbent_cost
#                 iterations_since_best = 0
           
#                 # New best solution: +4 points
#                 op_stats[chosen_op["name"]]["score"] += 4
               
#         elif feasibility:
#             if random.random() < 0.8:
#                 incumbent = new_sol.copy()
#                 incumbent_cost = new_cost
#             delta_w.append(delta_E)
            
#             # Feasible but not better: +1 point
#             op_stats[chosen_op["name"]]["score"] += 1
    
#     if delta_w:
#         delta_avg = np.mean(delta_w) 
#         T_0 = -delta_avg / math.log(0.8)
#     else:
#         T_0 = 1.0  
        
#     alpha = (T_f / T_0) ** (1/9900) 
#     T = T_0
    

#     for i in range(1, 9900):
#         # Checking if we need to escape local optimum
#         if iterations_since_best > escape_condition:
#             vehicle_ranges = zero_pos(incumbent)
            
#             for vehicle_index, (start, end) in enumerate(vehicle_ranges):
#                 if end > start:
#                     segment = list(incumbent[start:end])
#                     if len(segment) > 2:
#                         random.shuffle(segment)
#                         incumbent[start:end] = segment
#                     break
                
#             iterations_since_best = 0
#             incumbent_cost = cost_function(incumbent, prob)
            
#         # Updating operator probabilities periodically
#         if i % segment_size == 0 and i > 0:
#             total_score = sum(op_stats[op["name"]]["score"] for op in operators)
#             total_count = sum(op_stats[op["name"]]["counter"] for op in operators)
            
#             if total_score > 0 and total_count > 0:
#                 # Calculating new probabilities for each operator
#                 for op in operators:
#                     op_name = op["name"]
#                     counter = max(1, op_stats[op_name]["counter"])
#                     score = op_stats[op_name]["score"]
                    
#                     # Calculating normalized scores
#                     op_stats[op_name]["probability"] = (score / counter + 0.01) / (total_score / total_count + 0.03 * len(operators))
                
#                 # Normalize probabilities to sum to 1
#                 prob_sum = sum(op_stats[op["name"]]["probability"] for op in operators)
#                 if prob_sum > 0:
#                     for op in operators:
#                         op_stats[op["name"]]["probability"] /= prob_sum
            
#             # Reset scores and counters for the next segment
#             for op in operators:
#                 # print(f'The score of {op["name"]} = {op_stats[op["name"]]["score"]}')
#                 op_stats[op["name"]]["score"] = 0
#                 op_stats[op["name"]]["counter"] = 0
            
#         if i % 1000 == 0:
#             print(f"Iteration {i}, Best Cost: {best_cost}")
#             # print(f"Vi er her: {i}")
            
#         # Select and apply operator
#         probabilities = [op_stats[op["name"]]["probability"] for op in operators]
#         chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
#         chosen_op = operators[chosen_op_idx]
        
#         op_stats[chosen_op["name"]]["counter"] += 1
#         new_sol = chosen_op["function"](prob, incumbent)

#         feasibility, _ = feasibility_check(new_sol, prob)
#         if not feasibility:
#             continue
        
#         new_cost = cost_function(new_sol, prob)
#         delta_E = new_cost - incumbent_cost
       
#         if feasibility and delta_E < 0:
#             incumbent = new_sol.copy()
#             incumbent_cost = new_cost
            
#             # Better than incumbent: +2 points
#             op_stats[chosen_op["name"]]["score"] += 2
            
#             if incumbent_cost < best_cost:
#                 best_sol = incumbent.copy()
#                 best_cost = incumbent_cost
#                 iterations_since_best = 0
                
#                 # New best solution: +4 points
#                 op_stats[chosen_op["name"]]["score"] += 4
            
#             else:
#                 iterations_since_best += 1
                
#         elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
#             incumbent = new_sol.copy()
#             incumbent_cost = new_cost
            
#             iterations_since_best += 1
            
#             # Feasible but not better: +1 point
#             op_stats[chosen_op["name"]]["score"] += 1
                
#         else:
#             iterations_since_best += 1

#         T = alpha * T
    
#     return best_sol, op_stats


# IDEA:
# Make the reinsertion functions also weighted, such that the 'general_adaptive_metaheuristics' function can be sure to use the right reinsertion function on the best operator! 

# def general_adaptive_metaheuristics(prob, initial_sol, segment_size = 100):
    # best_sol = initial_sol.copy()
    # incumbent = initial_sol.copy()
    # T_f = 0.1  
    
    # # probabilities for operators
    # operators = ["P1", "P2", "P3"]
    # P1, P2, P3 = 1/3, 1/3, 1/3
    # probabilities = [P1, P2, P3]
    
    # # Scores:
    # OP1_score = 0
    # OP2_score = 0
    # OP3_score = 0
    
    # # Usage counter:
    # OP1_counter = 0
    # OP2_counter = 0
    # OP3_counter = 0
    
    # iterations_since_best = 0
    
    # escape_condition = 1000 # After this many iterations without improvement, apply escape

    # incumbent_cost = cost_function(incumbent, prob)
    # best_cost = incumbent_cost

    # delta_w = []
    
    # for w in range(1, 100): 
    #     chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
    
    #     if chosen_operator == 'P1':
    #         OP1_counter += 1
    #         new_sol = OP1(prob, incumbent)
    #     elif chosen_operator == 'P2':
    #         OP2_counter += 1
    #         new_sol = OP2(prob, incumbent)
    #     elif chosen_operator == 'P3':
    #         OP3_counter += 1
    #         new_sol = OP3(prob, incumbent)            
            
    #     feasibility, _ = feasibility_check(new_sol, prob)
    #     if not feasibility:
    #         continue
        
    #     new_cost = cost_function(new_sol, prob)
    #     delta_E = new_cost - incumbent_cost
        
    #     if feasibility and delta_E < 0:
    #         incumbent = new_sol
    #         incumbent_cost = new_cost
            
    #         #  hvis jeg finner en bedre_sol en current_sol gi OP?_score 2 poeng
    #         if chosen_operator == 'P1':
    #             OP1_score += 2
    #         elif chosen_operator == 'P2':
    #             OP2_score += 2
    #         elif chosen_operator == 'P3':
    #             OP3_score += 2            
            
    #         if incumbent_cost < best_cost:
    #             best_sol = incumbent
    #             best_cost = incumbent_cost
    #             iterations_since_best = 0
                
    #             # hvis jeg finner en new_best, så gi OP?_score 4 poeng
    #             if chosen_operator == 'P1':
    #                 OP1_score += 4
    #             elif chosen_operator == 'P2':
    #                 OP2_score += 2
    #             elif chosen_operator == 'P3':
    #                 OP3_score += 4
               
    #     elif feasibility:
    #         if random.random() < 0.8:
    #             incumbent = new_sol
    #             incumbent_cost = new_cost
    #         delta_w.append(delta_E)
            
    #         # hvis jeg finner en new_sol som ikke er funnet før, gi OP?_score 1 poeng
    #         if chosen_operator == 'P1':
    #             OP1_score += 1
    #         elif chosen_operator == 'P2':
    #             OP2_score += 1
    #         elif chosen_operator == 'P3':
    #             OP3_score += 1
    
    # delta_avg = np.mean(delta_w) 
    # T_0 = -delta_avg / math.log(0.8)

    # alpha = (T_f / T_0) ** (1/9900) 
    # T = T_0
    
    # # Her tror jeg at jeg skal oppdatere 'the weights' av operatorene. Altså prosentandelen for at de ulike kjører
    
    # for i in range(1, 9900):
    #     # We have reached local optimum, and have to escape
    #     if iterations_since_best > escape_condition:
    #         vehicle_ranges = zero_pos(incumbent)
            
    #         for vehicle_index, (start, end) in enumerate(vehicle_ranges):
    #             if end > start:
    #                 segment = incumbent[start:end]
    #                 if len(segment) > 2:
    #                     # np.random.shuffle(segment)
    #                     # incumbent[start:end] = segment
    #                     segment_list = list(segment)
    #                     random.shuffle(segment_list) 
    #                     incumbent[start:end] = segment_list
    #                 break
                
    #         iterations_since_best = 0
    #         incumbent_cost = cost_function(incumbent, prob)
            
    #         # Updating the weights according to the scores:
    #     if i % segment_size == 0 and i > 0: # HVA ER segment_size?
    #         total_score = OP1_score + OP2_score + OP3_score
    #         total_count = OP1_counter + OP2_counter + OP3_counter
    #         if total_score > 0:
    #             P1 = (OP1_score / max(1, OP1_counter) + 0.01) / (total_score / total_count + 0.03 * len(operators))
    #             P2 = (OP2_score / max(1, OP2_counter) + 0.01) / (total_score / total_count + 0.03 * len(operators))
    #             P3 = (OP3_score / max(1, OP3_counter) + 0.01) / (total_score / total_count + 0.03 * len(operators))
            
    #             weight_sum = P1 + P2 + P3
    #             P1 /= weight_sum
    #             P2 /= weight_sum
    #             P3 /= weight_sum
            
    #         # Reset scores and counters for the next segment
    #         OP1_score = 0
    #         OP2_score = 0
    #         OP3_score = 0
            
    #         OP1_counter = 0
    #         OP2_counter = 0
    #         OP3_counter = 0
            
    #         # # Print progress
    #         # if i % 1000 == 0:
    #         #     print(f"Iteration {i}, Best Cost: {best_cost}, Current Weights: {weights}")
            
    #     if i % 1000 == 0:
    #         print(f"Vi er her: {i}")
            
    #     chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
    #     if chosen_operator == 'P1':
    #         OP1_counter += 1
    #         new_sol = OP1(prob, incumbent)
    #     elif chosen_operator == 'P2':
    #         OP2_counter += 1
    #         new_sol = OP2(prob, incumbent)
    #     elif chosen_operator == 'P3':
    #         OP3_counter += 1
    #         new_sol = OP3(prob, incumbent)

    #     feasibility, _ = feasibility_check(new_sol, prob)
    #     if not feasibility:
    #         continue
        
    #     new_cost = cost_function(new_sol, prob)
    #     delta_E = new_cost - incumbent_cost
       
    #     if feasibility and delta_E < 0:
    #         incumbent = new_sol
    #         incumbent_cost = new_cost
            
    #         #  hvis jeg finner en bedre_sol enn current_sol gi OP?_score 2 poeng
    #         if chosen_operator == 'P1':
    #             OP1_score += 2
    #         elif chosen_operator == 'P2':
    #             OP2_score += 2
    #         elif chosen_operator == 'P3':
    #             OP3_score += 2   
            
            
    #         if incumbent_cost < best_cost:
    #             best_sol = incumbent
    #             best_cost = incumbent_cost
                
    #             iterations_since_best = 0
                
    #             # hvis jeg finner en new_best, så gi OP?_score 4 poeng
    #             if chosen_operator == 'P1':
    #                 OP1_score += 4
    #             elif chosen_operator == 'P2':
    #                 OP2_score += 2
    #             elif chosen_operator == 'P3':
    #                 OP3_score += 4
            
    #         else:
    #             iterations_since_best += 1
                
    #     elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
    #         incumbent = new_sol
    #         incumbent_cost = new_cost
            
    #         iterations_since_best += 1
            
    #         # hvis jeg finner en new_sol som ikke er funnet før, gi OP?_score 1 poeng
    #         if chosen_operator == 'P1':
    #             OP1_score += 1
    #         elif chosen_operator == 'P2':
    #             OP2_score += 1
    #         elif chosen_operator == 'P3':
    #             OP3_score += 1
                
    #     else:
    #         iterations_since_best += 1

    #     T = alpha * T
    
    # return best_sol, best_cost
    
            
        

























