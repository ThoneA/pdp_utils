# # # import numpy as np


# # # l = [1,2,3,4,0,1,2,0,3,4,5,0]
# # # x = l.index(0)
# # # print(x)
# # # for i in range(x+1, len(l)):
# # #     print(l[i])
# # # item = 2
# # # # print(l)
# # # l = list(filter((item).__ne__, l))

# # # print(l)
# # # call = [7,7]
# # # zero_count = 3
# # # rand_zero = 2
# # # for i in range(len(l)):
# # #     if l[i] == 0:
# # #         rand_zero -=1
# # #         zero_count -=1
# # #     if rand_zero == 0 and zero_count == 0:
# # #         pickup_index = np.random.randint(i + 1, len(l) + 1) 
# # #         l.insert(pickup_index, call[0])
# # #         delivery_index = np.random.randint(pickup_index + 1, len(l) + 1)
# # #         l.insert( delivery_index, call[1])
# # #         break
# # #     elif rand_zero == 0:
# # #         next_zero_pos = 0
# # #         for j in range(i+1, len(l) + 1):
# # #             if l[j] == 0:
# # #                 next_zero_pos = j
# # #                 break
# # #         print(next_zero_pos)
# # #         pickup_index = np.random.randint(i + 1, next_zero_pos + 1)
# # #         l.insert(pickup_index, call[0])
# # #         delivery_index = np.random.randint(pickup_index + 1, next_zero_pos + 2)
# # #         l.insert( delivery_index, call[1])
        
# # #         break
  
# # # l = [1,2,3]  
# # # # for i in range(len(l)):
# # # #     for j in range(i+1, len(l)):
# # # #         print(j)
    
# # # print(l)




# # # call = []
# # # call = [2] * 2




# # # l = [1,2,3,4,0,5,6,7,8,0,9,10,11,0,12,13,14]
# # # l = [1,2,0,3]

# # # next_zero_index = None
# # # for i in range(len(l)):
# # #     for j in range(i, len(l)):
# # #         if l[j] == 0:
# # #             next_zero_index = j
# # #             break
    
# # # print(next_zero_index)

# # from pdp_utils import *
# # prob = load_problem('pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt')


# # sol = [1,3,2,3,2,1,0,4,4,0,5,6,5,7,6,7,0]

# # # calls = sol[0:6]
# # calls = set(sol[0:6])
# # calls = list(calls)
# # calls = [x - 1 for x in calls]

# # print(calls) # De som er i bilen
# # # Calls bilen kan ta 
# # vehicle = 1
# # calls_for_vehicle = prob['VesselCargo'][vehicle - 1]
# # # calls_for_vehicle er binary, så omgjøre til index+1 hvis 1
# # calls_for_vehicle = [i + 1 for i, x in enumerate(calls_for_vehicle) if x == 1]
# # print(calls_for_vehicle)

# # # Calls som ikke er i bilen
# # calls_not_in_vehicle = [x for x in calls_for_vehicle if x not in calls]
# # print(calls_not_in_vehicle)
# # # Minste size av call som ikke er i bilen
# # dict = {}
# # for call in calls_not_in_vehicle:
# #     dict[call] = prob['Cargo'][call - 1, 2]

# # print(dict)

# # vehicle_weight = sum(prob['Cargo'][calls, 2])
# # print(vehicle_weight)

# # capacity = prob['VesselCapacity'][0]

# # smallest_possible_call = min(dict, key=dict.get)
# # print(smallest_possible_call)


# # # Så lenge det ikke er mer plass i bilen til en ny call så skal bilen bli valgt
# # if capacity - vehicle_weight < dict[smallest_possible_call]:
# #     chosen_vehicle = vehicle
# #     print("capacity: " , capacity)
# #     print("vehicle_weight: ", vehicle_weight)
# #     print("diff: ", capacity - vehicle_weight)
# #     print("Chosen vehicle: ", chosen_vehicle)
# # # else:
# # #     continue

# # new_sol = sol.copy()


# # possible_calls_to_reinsert = calls_for_vehicle
# # print(possible_calls_to_reinsert)
# # calls_to_reinsert = []

# # new_sol = [x for x in new_sol if x not in calls_to_reinsert]

# # num_to_select = random.randint(1, len(possible_calls_to_reinsert))

# # calls_to_reinsert = random.sample(possible_calls_to_reinsert, num_to_select)

# # new_sol = [x for x in new_sol if x not in calls_to_reinsert]
# # new_sol = new_sol + calls_to_reinsert + [8]

# # print(calls_to_reinsert)


# # def zero_pos(sol):
# #     zero_pos = [i for i, x in enumerate(sol) if x == 0]
# #     vehicle_ranges = []
# #     start_index = 0  
    
# #     # Defines vehicle ranges in the solution
# #     for zero in zero_pos:
# #         vehicle_ranges.append((start_index, zero)) 
# #         start_index = zero + 1
# #     vehicle_ranges.append((start_index, len(sol) - 1))
    
# #     return vehicle_ranges

# # print(new_sol)
# # vehicle_ranges = zero_pos(new_sol)
# # print(vehicle_ranges)

# import random
# import numpy as np


# # n = np.random.randint(0, 3)

# # while n != 3:
# #     n = np.random.randint(0, 3)

# # # print(n)

# vehicle_ranges = ((0,0), (1,1), (2,2), (3,7))

# start, end = list(vehicle_ranges)[1]

# # vehicle_index, (start, end) = list(enumerate(vehicle_ranges))[-1]
# print(start, end)

# # if end > start:
# #     print('hei')
# #     print(end, start)
# # x = "123"
# # chosen_reinsertion = random.choice(x)

# # if chosen_reinsertion == "1":
# #     print("1 er valgt")
# # elif chosen_reinsertion == "2":
# #     print("2 er valgt")
# # elif chosen_reinsertion == "3":
# #     print("3 er valgt")
# # else:
# #     print("dette fungerer ikke")
    
    
    
    
    
    
    
    
# import time
# from pdp_utils import *
# from pdp_utils.operators import *
# import traceback
    
    
    
    
    
# def greedy_reinsert(calls, prob, removed_sol): # KANSKJE KUNN SJEKKE HALVPARTEN AV BILENE VELG DEM RANDOM
#     best_sol = removed_sol.copy()
    
#     for call in calls:
#         vehicle_ranges = zero_pos(removed_sol)
#         new_best_sol = best_sol.copy()
#         new_best_cost = 1e12
        
#         for vehicle_index, (start, end) in enumerate(vehicle_ranges):
#             if vehicle_index == prob['n_vehicles']:
#                 continue
#             if prob['VesselCargo'][vehicle_index][call - 1] == 0:
#                 continue
            
#             # PICKUP
#             for p_pos in range(start, end + 1):
#                 temp_p_sol = best_sol.copy()
#                 temp_p_sol.insert(p_pos, call)
                
#                 # DELIVERY             
#                 for d_pos in range(p_pos + 1, end + 2):
#                     temp_d_sol = temp_p_sol.copy()
#                     temp_d_sol.insert(d_pos, call)
                
#                     feasibility, _ = feasibility_check(temp_d_sol, prob)
#                     if feasibility:
#                         temp_cost = cost_function(temp_d_sol, prob)
                        
#                         if temp_cost < new_best_cost:
#                             new_best_sol = temp_d_sol
#                             new_best_cost = temp_cost
                            
#         if new_best_cost == 1e12:
#             best_sol.insert(len(best_sol), call)
#             best_sol.insert(len(best_sol), call)
#         else:
#             best_sol = new_best_sol
        
#     return best_sol


# def OP2(prob, sol, reinsert):
#     new_sol = sol.copy()
#     calls = prob['n_calls']
#     calls_to_reinsert = []

#     # Choose a random number between 1 and 10
#     if calls < 10:
#         calls_n = np.random.randint(1, calls + 1)
#     else:
#         calls_n = np.random.randint(2, 10)
    
#     calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)

#     # Remove selected calls
#     new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
#     new_sol = reinsert(calls_to_reinsert, prob, new_sol)
    
#     return new_sol

# prob = load_problem('pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt')
# initial_sol = initial_solution(prob)

# print(OP2(prob, initial_sol, greedy_reinsert))


# if not False:
#     print('hei')

# import numpy as np

# possible_vehicles = [1, 6, 7]

# selected_vehicles = np.random.choice(possible_vehicles, size = 3,  replace=False)

# print(selected_vehicles)

# selected_vehicles = list(selected_vehicles)

# for i in selected_vehicles:
#     print(i)





# new_cost = cost_function(new_sol, prob)
# delta_E = new_cost - incumbent_cost
# # print(f"incumbent: {incumbent}, new: {incumbent_cost}, delta_e: {delta_E}")
# # print(f"new_sol: {new_sol}, new: {new_cost}, delta_e: {delta_E}")
# # print(f"delta_e: {delta_E}")

# g = i
# G = total_iterations
# D = 0.2 * ((G-g)/G) * best_cost
# max_acceptable_cost = best_cost + D

# if feasibility and delta_E < 0:
#     incumbent = new_sol.copy()
#     incumbent_cost = new_cost
    
#     # Better than incumbent: +2 points
#     op_stats[chosen_op["name"]]["score"] += 2
    
#     if incumbent_cost < best_cost:
#         best_sol = incumbent.copy()
#         best_cost = incumbent_cost
#         iterations_since_best = 0
        
#         # New best solution: +4 points
#         op_stats[chosen_op["name"]]["score"] += 4
        
#         # Record the iteration where we found a better solution
#         history["iterations"].append(current_iteration)
#         history["best_costs"].append(best_cost)
#     else:
#         iterations_since_best += 1
        
# # elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
# elif feasibility and (new_cost <= max_acceptable_cost):
#     incumbent = new_sol.copy()
#     incumbent_cost = new_cost
#     iterations_since_best += 1
    
#     # Feasible but not better: +1 point
#     op_stats[chosen_op["name"]]["score"] += 1
        
# else:
    # iterations_since_best += 1
    
i = 0

if i != 0 and i % 500 == 0:
    print("True")
else:
    print("False")
