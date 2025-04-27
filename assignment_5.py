import math
import random
from pdp_utils.Utils import *
import matplotlib.pyplot as plt
import numpy as np


"""
This function returns the vehicle ranges 
"""
def zero_pos(sol):
    sol_array = np.array(sol)
    zero_indices = np.where(sol_array == 0)[0]
    vehicle_ranges = []
    start_index = 0  
    
    for zero in zero_indices:
        vehicle_ranges.append((start_index, zero)) 
        start_index = zero + 1
    vehicle_ranges.append((start_index, len(sol) - 1))
    
    return vehicle_ranges

"""
This reinsertion function chooses a random vehicle, then finds the best position to place the pickup and delivery for all calls
"""
def soft_greedy_reinsert(calls, prob, removed_sol):
    
    best_sol = removed_sol
    vehicle_ranges = zero_pos(removed_sol)
    vehicles_n = prob['n_vehicles']
    # print(f"soft greedy gets:")
    # print(f" calls to reinsert: {calls}, removed_solution {removed_sol}")
    
    
    for call in calls:        
        selected_vehicle = np.random.randint(0, vehicles_n)
        while prob['VesselCargo'][selected_vehicle][call - 1] == 0:
            selected_vehicle = np.random.randint(0, vehicles_n)
        
        # print(f"index of vehicle: {selected_vehicle}")
        
        start, end = list(vehicle_ranges)[selected_vehicle]
        new_best_sol = best_sol.copy()
        new_best_cost = 1e12
        
 
        # PICKUP
        for p_pos in range(start, end + 1):
            temp_p_sol = best_sol.copy()
            temp_p_sol.insert(p_pos, call)
            
            # DELIVERY
            for d_pos in range(p_pos + 1, end + 2):
                temp_d_sol = temp_p_sol.copy()
                temp_d_sol.insert(d_pos, call)
                
                feasibility, _ = feasibility_check(temp_d_sol, prob)
                
                if feasibility:
                    temp_cost = cost_function(temp_d_sol, prob)
                    
                    if temp_cost < new_best_cost:
                        new_best_sol = temp_d_sol
                        new_best_cost = temp_cost
        
        if call in new_best_sol:
            best_sol = new_best_sol.copy()

        # Dummy reinsertion
        else:
            best_sol.insert(len(best_sol), call)
            best_sol.insert(len(best_sol), call)           
        
    # print(f" new solution: {new_best_sol}")
    return best_sol
              
"""
This reinsertion function chooses a random vehicle, then finds the best position to place the pickup and delivery for all calls
"""
def soft_greedy_reinsert_2(calls, prob, removed_sol):
    best_sol = removed_sol
    vehicle_ranges = zero_pos(removed_sol)
    vehicles_n = prob['n_vehicles']
    # print(f"soft greedy gets:")
    # print(f" calls to reinsert: {calls}, removed_solution {removed_sol}")
    
    
    for call in calls:
        # Choose 3 different vehicles to find the best posision
        # Sjekk hvor mange vehicles som kan ta gitt call, hvis det er 3 eller mer så velg 3 random
        possible_vehicles = np.where(prob['VesselCargo'][:, call - 1] == 1)[0]
        if len(possible_vehicles) >= 3:
            selected_vehicle = np.random.choice(possible_vehicles, size=3, replace=False)
        else:
            selected_vehicle = np.random.choice(possible_vehicles, replace=False)
          
        # selected_vehicle = np.random.randint(0, vehicles_n)
        # while prob['VesselCargo'][selected_vehicle][call - 1] == 0:
        #     selected_vehicle = np.random.randint(0, vehicles_n)
        
        # print(f"index of vehicle: {selected_vehicle}")
        new_best_sol = best_sol.copy()
        new_best_cost = 1e12
        if isinstance(selected_vehicle, np.int64):
            selected_vehicle = [selected_vehicle]
        else:
            selected_vehicle = list(selected_vehicle)
        
        for vehicle in selected_vehicle:
            start, end = list(vehicle_ranges)[vehicle]
            
        
        # start, end = list(vehicle_ranges)[selected_vehicle]
        # new_best_sol = best_sol.copy()
        # new_best_cost = 1e12
        
 
        # PICKUP
            for p_pos in range(start, end + 1):
                temp_p_sol = best_sol.copy()
                temp_p_sol.insert(p_pos, call)
                
                # DELIVERY
                for d_pos in range(p_pos + 1, end + 2):
                    temp_d_sol = temp_p_sol.copy()
                    temp_d_sol.insert(d_pos, call)
                    
                    feasibility, _ = feasibility_check(temp_d_sol, prob)
                    
                    if feasibility:
                        temp_cost = cost_function(temp_d_sol, prob)
                        
                        if temp_cost < new_best_cost:
                            new_best_sol = temp_d_sol
                            new_best_cost = temp_cost
            
        if call in new_best_sol:
            best_sol = new_best_sol.copy()

        # Dummy reinsertion
        else:
            best_sol.insert(len(best_sol), call)
            best_sol.insert(len(best_sol), call)           
            
    # print(f" new solution: {new_best_sol}")
    return best_sol   
        

"""
This reinsertion function checks the two best positions for the pickup and delivery of the calls, then it checks the difference between the cost of the two solution for every call, and then chooses the one that has the smallest difference.
"""
# def k_regret(calls, prob, removed_sol, k = 2):
#     best_sol = removed_sol
#     feasibility_cache = {}
#     cost_cache = {}
#     vehicles_n = prob['n_vehicles']
#     remaining_calls = set(calls)
        
#     while remaining_calls:
#         k_calls = []
#         vehicle_ranges = zero_pos(best_sol)
#         new_calls = []
#         for call in calls:
#             if call in remaining_calls:
#                 new_calls.append(call)
        
#         for call in new_calls:
#             if call not in remaining_calls:
#                 continue
#             k_best_positions = []
#             for vehicle_index, (start, end) in enumerate(vehicle_ranges):
#                 if vehicle_index >= vehicles_n:
#                     continue
#                 # Check if the vehicle can take the call
#                 print(f"call: {call}, vehicle_index: {vehicle_index}, start: {start}, end: {end}")
#                 if prob['VesselCargo'][vehicle_index][call - 1] == 0:
#                     continue
                
#                 # Find the k best positions for the pickup and delivery of the call
                
#                 # PICKUP
#                 for p_pos in range(start, end + 1):
#                     temp_p_sol = best_sol.copy()
#                     temp_p_sol.insert(p_pos, call)
                    
#                     # DELIVERY
#                     for d_pos in range(p_pos + 1, end + 2):
#                         temp_d_sol = temp_p_sol.copy()
#                         temp_d_sol.insert(d_pos, call)
                        
#                         sol_key = tuple(temp_d_sol)
                        
#                         if sol_key in feasibility_cache:
#                             feasibility, _ = feasibility_cache[sol_key]
#                         else:
#                             feasibility, _ = feasibility_check(temp_d_sol, prob)
#                             feasibility_cache[sol_key] = (feasibility, _)
                        
#                         if feasibility:
#                             if sol_key in cost_cache:
#                                 temp_cost = cost_cache[sol_key]
#                             else:
#                                 temp_cost = cost_function(temp_d_sol, prob)
#                                 cost_cache[sol_key] = temp_cost
                                
#                             k_best_positions.append((temp_d_sol, temp_cost))
                    
#                 # Choose the k best posistions according to the cost
#             k_best_positions = sorted(k_best_positions, key=lambda x: x[1])[:k]  
#             k_calls.append(k_best_positions)
        
#         # Check the difference between the cost of the k solution for every call
#         cost_for_calls_diffs = []
#         for i in range(len(k_calls)):
#             k_best_positions = k_calls[i]
#             if len(k_best_positions) < k:
#                 continue
            
#             # Calculate the difference between the cost of the two solution
#             end = k-1
#             cost_diff = abs(k_best_positions[0][1] - k_best_positions[end][1])
            
#             cost_for_calls_diffs.append((new_calls[i], cost_diff))
            
#         # Sort the calls according to the cost difference, from biggest to smallest
#         cost_for_calls_diffs = sorted(cost_for_calls_diffs, key=lambda x: x[1], reverse=True)
        
#         if not cost_for_calls_diffs:
#             break
#         biggest_diff_call = cost_for_calls_diffs[0][0]
#         new_best_sol = best_sol.copy()
#         new_best_cost = 1e12
        
#         # Take the first call from the cost_for_calls_diffs list, and insert it into the best solution
#         for vehicle_index, (start, end) in enumerate(vehicle_ranges):
#                 if vehicle_index >= vehicles_n:
#                     continue
#                 # Check if the vehicle can take the call
#                 if prob['VesselCargo'][vehicle_index][biggest_diff_call - 1] == 0:
#                     continue
                
#                 # Find the k best positions for the pickup and delivery of the call
                
#                 # PICKUP
#                 for p_pos in range(start, end + 1):
#                     temp_p_sol = best_sol.copy()
#                     temp_p_sol.insert(p_pos, biggest_diff_call)
                    
#                     # DELIVERY
#                     for d_pos in range(p_pos + 1, end + 2):
#                         temp_d_sol = temp_p_sol.copy()
#                         temp_d_sol.insert(d_pos, biggest_diff_call)
                        
#                         sol_key = tuple(temp_d_sol)
                        
#                         if sol_key in feasibility_cache:
#                             feasibility, _ = feasibility_cache[sol_key]
#                         else:
#                             feasibility, _ = feasibility_check(temp_d_sol, prob)
#                             feasibility_cache[sol_key] = (feasibility, _)
                        
#                         if feasibility:
#                             if sol_key in cost_cache:
#                                 temp_cost = cost_cache[sol_key]
#                             else:
#                                 temp_cost = cost_function(temp_d_sol, prob)
#                                 cost_cache[sol_key] = temp_cost
                                
#                             if temp_cost < new_best_cost:
#                                 new_best_sol = temp_d_sol
#                                 new_best_cost = temp_cost
                                
#         best_sol = new_best_sol.copy()
#         # calls = np.delete(calls, np.where(calls == biggest_diff_call))
#         remaining_calls.remove(biggest_diff_call)
    
#     return best_sol         
            
def k_regret(calls, prob, removed_sol, k=2):
    best_sol = removed_sol
    feasibility_cache = {}
    cost_cache = {}
    vehicles_n = prob['n_vehicles']
    remaining_calls = set(calls)
        
    while remaining_calls:
        k_calls = []
        vehicle_ranges = zero_pos(best_sol)
        new_calls = [call for call in calls if call in remaining_calls]
        
        for call in new_calls:
            # print(f"call: {call}, remaining_calls: {remaining_calls}")
            k_best_positions = []
            for vehicle_index, (start, end) in enumerate(vehicle_ranges):
                if vehicle_index >= vehicles_n:
                    continue
                # Check if the vehicle can take the call
                if prob['VesselCargo'][vehicle_index][call - 1] == 0:
                    continue
                
                # Find the k best positions for the pickup and delivery of the call
                for p_pos in range(start, end + 1):
                    temp_p_sol = best_sol.copy()
                    temp_p_sol.insert(p_pos, call)
                    
                    for d_pos in range(p_pos + 1, end + 2):
                        temp_d_sol = temp_p_sol.copy()
                        temp_d_sol.insert(d_pos, call)
                        
                        sol_key = tuple(temp_d_sol)
                        
                        if sol_key in feasibility_cache:
                            feasibility, _ = feasibility_cache[sol_key]
                        else:
                            feasibility, _ = feasibility_check(temp_d_sol, prob)
                            feasibility_cache[sol_key] = (feasibility, _)
                        
                        if feasibility:
                            # print(f"feasibility: {feasibility}, sol_key: {sol_key}")
                            if sol_key in cost_cache:
                                temp_cost = cost_cache[sol_key]
                            else:
                                temp_cost = cost_function(temp_d_sol, prob)
                                cost_cache[sol_key] = temp_cost
                                
                            k_best_positions.append((temp_d_sol, temp_cost))
                            # print(f"k_best_positions: {k_best_positions}")
            
            # Keep only the k best positions
            k_best_positions = sorted(k_best_positions, key=lambda x: x[1])[:k]
            if len(k_best_positions) >= k:
                k_calls.append(k_best_positions)
        
        # Check the difference between the cost of the k solutions for every call
        cost_for_calls_diffs = []
        for i, k_best_positions in enumerate(k_calls):
            if len(k_best_positions) < k:
                continue
            
            # Calculate the difference between the cost of the two solutions
            cost_diff = abs(k_best_positions[0][1] - k_best_positions[k - 1][1])
            cost_for_calls_diffs.append((new_calls[i], cost_diff))
        
        # Sort the calls according to the cost difference, from biggest to smallest
        cost_for_calls_diffs = sorted(cost_for_calls_diffs, key=lambda x: x[1], reverse=True)
        
        if not cost_for_calls_diffs:
            break
        
        # Take the call with the biggest cost difference
        biggest_diff_call = cost_for_calls_diffs[0][0]
        new_best_sol = best_sol.copy()
        new_best_cost = float('inf')
        
        # Insert the call into the best solution
        for vehicle_index, (start, end) in enumerate(vehicle_ranges):
            if vehicle_index >= vehicles_n:
                continue
            if prob['VesselCargo'][vehicle_index][biggest_diff_call - 1] == 0:
                continue
            
            for p_pos in range(start, end + 1):
                temp_p_sol = best_sol.copy()
                temp_p_sol.insert(p_pos, biggest_diff_call)
                
                for d_pos in range(p_pos + 1, end + 2):
                    temp_d_sol = temp_p_sol.copy()
                    temp_d_sol.insert(d_pos, biggest_diff_call)
                    
                    sol_key = tuple(temp_d_sol)
                    
                    if sol_key in feasibility_cache:
                        feasibility, _ = feasibility_cache[sol_key]
                    else:
                        feasibility, _ = feasibility_check(temp_d_sol, prob)
                        feasibility_cache[sol_key] = (feasibility, _)
                    
                    if feasibility:
                        if sol_key in cost_cache:
                            temp_cost = cost_cache[sol_key]
                        else:
                            temp_cost = cost_function(temp_d_sol, prob)
                            cost_cache[sol_key] = temp_cost
                            
                        if temp_cost < new_best_cost:
                            new_best_sol = temp_d_sol
                            new_best_cost = temp_cost
        
        if biggest_diff_call in new_best_sol:
            best_sol = new_best_sol.copy()

        # Dummy reinsertion
        else:
            best_sol.insert(len(best_sol), biggest_diff_call)
            best_sol.insert(len(best_sol), biggest_diff_call)
            
        remaining_calls.remove(biggest_diff_call)
    
    if remaining_calls:
        for call in remaining_calls:
            best_sol.insert(len(best_sol), call)
            best_sol.insert(len(best_sol), call)
    
    return best_sol
        
        

"""
This reinsertion function, chooses a vehicle randomly and then adds the calls one by one into random vehicles
"""
def random_reinsert(calls, prob, removed_sol):
    vehicles_n = prob['n_vehicles']
    vehicle_ranges = zero_pos(removed_sol)
    new_sol = removed_sol
    
    for call in calls:    
        vehicle_to_select = np.random.randint(0, vehicles_n)
        vehicle_index, (start, end) = list(enumerate(vehicle_ranges))[vehicle_to_select]
        
        # Insert pickup
        pickup_pos = np.random.randint(start, end + 1)
        new_sol.insert(pickup_pos, call)
        
        # Insert delivery
        if pickup_pos < end:
            delivery_pos = np.random.randint(pickup_pos + 1, end + 1)
        else:
            delivery_pos = pickup_pos + 1
        new_sol.insert(delivery_pos, call)
    
    return new_sol

"""
This insertion function inserts all the calls into a random vehicle
"""
def easy_reinsert(calls, prob, removed_sol):
    vehicles_n = prob['n_vehicles']
    vehicle_ranges = zero_pos(removed_sol)
    vehicle_to_select = np.random.randint(0, vehicles_n)
    i_vehicle = vehicle_ranges[vehicle_to_select][0]
    
    new_sol = removed_sol
    i_pos = 0
      
    for call in calls:
        new_sol.insert(i_vehicle + i_pos, call)
        i_pos += 1
        new_sol.insert(i_vehicle + i_pos, call)
        i_pos += 1 
 
    return new_sol
 
"""
This insertion function inserts all the calls into a random vehicle, then shuffles the calls in that vehicle
"""   
def easy_shuffle_reinsert(calls, prob, removed_sol):
    vehicles_n = prob['n_vehicles']
    vehicle_ranges = zero_pos(removed_sol)
    vehicle_to_select = np.random.randint(0, vehicles_n)
    i_vehicle = vehicle_ranges[vehicle_to_select][0]

    new_sol = removed_sol
    i_pos = 0

    for call in calls:
        new_sol.insert(i_vehicle + i_pos, call)
        i_pos += 1
        new_sol.insert(i_vehicle + i_pos, call)
        i_pos += 1

    # Shuffle the calls:
    vehicle_ranges = zero_pos(new_sol)
    start, end = vehicle_ranges[vehicle_to_select]

    shuffled_part = np.array(new_sol[start:end+1])
    shuffled_part = shuffled_part[:-1]  
    np.random.shuffle(shuffled_part)

    new_sol[start:end+1] = shuffled_part.tolist() + [0]
    
    return new_sol

# Ha en reinsert som sjekker om noen biler ikke har noen calls, for så og prøv å reinsert i den
def empty_reinsert(calls, prob, removed_sol):
    new_sol = removed_sol
    vehicle_ranges = zero_pos(new_sol)
    
    for call in calls:
        inserted_calls = False
        for vehicle_index, (start, end) in enumerate(vehicle_ranges):
            if end > start:
                continue
            else:
                temp_sol = new_sol.copy()
                temp_sol.insert(start + 1, call)
                temp_sol.insert(start + 2, call)
                feasibility, _ = feasibility_check(temp_sol, prob)
                
                if feasibility:
                    new_sol = temp_sol
                    inserted_calls = True
                    break
        
        if not inserted_calls:
            new_sol.append(call)
            new_sol.append(call)
    
    return new_sol   
        
"""
This operator chooses the vehicle with the biggest weight, and then chooses
a random number of calls between one and ten of the calls inside that vehicle. 
# """
def weighted_removal(prob, sol, reinsert): 
    new_sol = sol.copy()
    vehicle_ranges = zero_pos(sol)
    biggest_weight = 0
    calls_to_reinsert = [] 
    calls = prob['n_calls']
    
    for vehicle_index, (start, end) in enumerate(vehicle_ranges):      
        vehicle_calls = new_sol[start:end]
        unique_calls = set(vehicle_calls)
        unique_calls.discard(0)
        
        if not unique_calls:
            continue
    
        calls_list = list(unique_calls)
        call_indices = np.array([x - 1 for x in calls_list if x - 1 < calls])
        
        if len(call_indices) > 0:
            vehicle_weight = np.sum(prob['Cargo'][call_indices, 2])
            
            if vehicle_weight > biggest_weight:
                biggest_weight = vehicle_weight
                calls_to_reinsert = calls_list
    
    if calls_to_reinsert:
        if len(calls_to_reinsert) < 10:
            num_to_select = np.random.randint(1, len(calls_to_reinsert) + 1)
        else:
            num_to_select = np.random.randint(1, 10)
        
        selected_calls = np.random.choice(calls_to_reinsert, num_to_select, replace=False)
        
        # Remove selected calls
        new_sol = [x for x in new_sol if x not in selected_calls]
        
        # x = "1234"
        # chosen_reinsertion = random.choice(x)
        
        # if chosen_reinsertion == "1":
        #     new_sol = random_reinsert(selected_calls, prob, new_sol)
        # elif chosen_reinsertion == "2":
        #     new_sol = easy_reinsert(selected_calls, prob, new_sol)
        # elif chosen_reinsertion == "3":
        #     new_sol = easy_shuffle_reinsert(selected_calls, prob, new_sol)
        # elif chosen_reinsertion == "4":
        #     new_sol = soft_greedy_reinsert(selected_calls, prob, new_sol)
            # new_sol = empty_reinsert(selected_calls, prob, new_sol)
        
        new_sol = reinsert(selected_calls, prob, new_sol)
      
    return new_sol

"""
This operator randomly chooses between 1 and 10 calls depending on the size of the file.
Then inserst the calls back into the solution by using a easy_shuffle_reinsert.
"""
def random_removal_1(prob, sol, reinsert):
    new_sol = sol.copy()
    calls = prob['n_calls']
    calls_to_reinsert = []

    # Choose a random number between 1 and 10
    if calls < 10:
        calls_n = np.random.randint(1, calls + 1)
    else:
        calls_n = np.random.randint(2, 10)
    
    calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)
    # print(f"calls to remove: {calls_to_reinsert}")
    

    # Remove selected calls
    new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    # print(f"solution without the calls: {new_sol}")
    
    # new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    # new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
    # x = "1234"
    # chosen_reinsertion = random.choice(x)
    
    # if chosen_reinsertion == "1":
    #     new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "2":
    #     new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "3":
    #     new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "4":
    #     new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
        # new_sol = empty_reinsert(calls_to_reinsert, prob, new_sol)
    
    new_sol = reinsert(calls_to_reinsert, prob, new_sol)
        
    return new_sol

"""
This operator chooses randomly a car that contains calls,
then it randomly chooses calls between 1 and 10.
"""
def random_removal_2(prob, sol, reinsert):
    new_sol = sol.copy()
    vehicles = prob['n_vehicles']
    vehicle_ranges = zero_pos(sol)
    
    chosen_vehicle_index = np.random.randint(0, vehicles + 1)

    while vehicle_ranges[chosen_vehicle_index][0] == vehicle_ranges[chosen_vehicle_index][1]:
        chosen_vehicle_index = np.random.randint(0, vehicles + 1)
    
    start, end = vehicle_ranges[chosen_vehicle_index]

    vehicle_calls = new_sol[start:end]
    unique_calls = set(vehicle_calls)
    unique_calls.discard(0)    
    calls_list = list(unique_calls)

    if len(calls_list) < 10:
        calls_n = np.random.randint(1, len(calls_list) + 1)
    elif len(calls_list) >= 10:
        calls_n = np.random.randint(2, 10)
    
    calls_to_reinsert = []
    while calls_n > 0:
        call = np.random.choice(calls_list)
        calls_list.remove(call)
        calls_n -= 1
        new_sol.remove(call)
        new_sol.remove(call)
        calls_to_reinsert.append(call)
    
    # new_sol = greedy_reinsert(calls_to_reinsert, prob, new_sol)
    # new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
    # x = "1234"
    # chosen_reinsertion = random.choice(x)
    
    # if chosen_reinsertion == "1":
    #     new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "2":
    #     new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "3":
    #     new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "4":
    #     new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
        # new_sol = empty_reinsert(calls_to_reinsert, prob, new_sol)
    
    new_sol = reinsert(calls_to_reinsert, prob, new_sol)
    
    return new_sol

"""
This operator checks if there are any calls in the dummy that can be inserted into any of the vehicles
"""
def dummy_removal(prob, sol, reinsert):
    new_sol = sol.copy()
    vehicle_ranges = zero_pos(new_sol)
    
    # The dummy
    vehicle_index, (start, end) = list(enumerate(vehicle_ranges))[-1]
    vehicle_calls = new_sol[start:end]
    unique_calls = set(vehicle_calls)
    unique_calls.discard(0)
    calls_list = list(unique_calls)
    
    if end > start:
        if len(calls_list) < 10:
            calls_n = np.random.randint(1, len(calls_list) + 1)
        elif len(calls_list) >= 10:
            calls_n = np.random.randint(2, 10)
    
            
        # calls_to_reinsert = random.sample(calls_list, calls_n)  
    # calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)
    calls_to_reinsert = random.sample(range(1, len(calls_list) + 1), calls_n)
    
    # Remove selected calls
    new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
    # new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
    # new_sol = empty_reinsert(calls_to_reinsert, prob, new_sol)
    # return new_sol
    
  
    # x = "1234"
    # chosen_reinsertion = random.choice(x)
    
    # if chosen_reinsertion == "1":
    #     new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "2":
    #     new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "3":
    #     new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
    # elif chosen_reinsertion == "4":
    #     new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    
    # new_sol = empty_reinsert(calls_to_reinsert, prob, new_sol)
    new_sol = reinsert(calls_to_reinsert, prob, new_sol)
    
    return new_sol 

def OP1(prob, sol):
    # new_sol =  random_removal_1(prob, sol, soft_greedy_reinsert_2)
    new_sol =  random_removal_1(prob, sol, k_regret)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = random_removal_1(prob, sol, soft_greedy_reinsert_2)
    # print(f"new_sol {new_sol}, used random removal 1")
    return new_sol

def OP2(prob, sol):
    # new_sol = random_removal_2(prob, sol, soft_greedy_reinsert_2)
    new_sol = random_removal_2(prob, sol, k_regret)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = random_removal_2(prob, sol, soft_greedy_reinsert_2)
    # print(f"new_sol {new_sol}, used random removal 2")
    return new_sol

def OP3(prob, sol):
    # new_sol = dummy_removal(prob, sol, soft_greedy_reinsert_2)
    new_sol = dummy_removal(prob, sol, k_regret)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = dummy_removal(prob, sol, soft_greedy_reinsert_2)
    # print(f"new_sol {new_sol}, used dummy removal")
    return new_sol

def OP4(prob, sol):
    # new_sol = weighted_removal(prob, sol, soft_greedy_reinsert_2)
    new_sol = weighted_removal(prob, sol, k_regret)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = weighted_removal(prob, sol, soft_greedy_reinsert_2)
    # print(f"new_sol {new_sol}, used weighted removal")
    return new_sol



def escape_algorithm(prob, incumbent, incumbent_cost, best_sol, best_cost, i_since_best):
    # for i in range(n_calls * 3): # KANSKJE HOPPE LITT LENGRE ENN 20?
    for i in range(20):
        if i == 0:
            print(f'No improvement for {i_since_best} iterations. Current best cost: {best_cost}, and current cost: {incumbent_cost}')
            print(f"Trying to escape local optimum...")
        new_sol = OP1(prob, incumbent)
        
        feasibility, _ = feasibility_check(new_sol, prob)
        if not feasibility:
            continue
        
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
        
        incumbent = new_sol.copy()
        incumbent_cost = new_cost
        
        if new_cost < best_cost:
            best_sol = new_sol.copy()
            best_cost = new_cost
            i_since_best = 0
            print(f"Found a better solution: {best_cost} at iteration {i}")
            break

    return incumbent, incumbent_cost, best_sol, best_cost, i_since_best
   
def acceptance_probability(new_sol, incumbent, incumbent_cost, i, total_iterations, best_cost, new_cost, feasibility, delta_E):
    g = i
    G = total_iterations
    D = 0.2 * ((G-g)/G) * best_cost
    max_acceptable_cost = best_cost + D
    score = 0
    
    # If I find a better solution than the current solution
    if feasibility and delta_E < 0:
        incumbent = new_sol.copy()
        incumbent_cost = new_cost
        score += 2
        
        # if incumbent_cost < best_cost:
        #     best_sol = incumbent.copy()
        #     best_cost = incumbent_cost
        #     i_since_best = 0
        #     score += 4
        
    elif feasibility and (new_cost <= max_acceptable_cost):
        incumbent = new_sol.copy()
        incumbent_cost = new_cost
        score += 1 # SKAL JEG HA DENNE????

    return incumbent, incumbent_cost, score
        
     

def general_adaptive_metaheuristics_2(prob, initial_sol, plot_results = True):
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    incumbent_cost = cost_function(incumbent, prob)
    best_cost = incumbent_cost
    i_since_best = 0
    total_iterations = 10000
    counter = 0

    operators = [
        {"name": "P1", "function": OP1},
        {"name": "P2", "function": OP2},
        {"name": "P3", "function": OP3},
        {"name": "P4", "function": OP4}
        # {"name": "P5", "function": OP5},
        # {"name": "P6", "function": OP6},
        # {"name": "P7", "function": OP7},
        # {"name": "P8", "function": OP8},
        # {"name": "P9", "function": OP9}
    ]
    
    op_stats = {}
    for op in operators:
        op_stats[op["name"]] = {
            "score": 0,
            "counter": 0,
            "probability": 1.0 / len(operators)
        }
        
    history = {
        "iterations": [],
        "best_costs": [],
        "operator_probs": {op["name"]: [] for op in operators},
        "operator_scores": {op["name"]: [] for op in operators},
        "segment_boundaries": []
    }
    
    
    # Hvilke stop condition skal jeg ha?
    for i in range(total_iterations):
        # print(f"Iteration {i}, Best Cost: {best_cost}")
        if i % 1000 == 0:
            print(f"********Iteration {i}, Best Cost: {best_cost}********") 
        
        if i_since_best != 0 and i_since_best % 500 == 0:
            incumbent, incumbent_cost, best_sol, best_cost, i_since_best = escape_algorithm(prob, incumbent, incumbent_cost, best_sol, best_cost, i_since_best)
        
        new_sol = incumbent.copy()
        
        # Choose a operator depending on selection parameters to apply to new_sol
        probabilities = [op_stats[op["name"]]["probability"] for op in operators]
        chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
        chosen_op = operators[chosen_op_idx]
        
        # Apply selected operator
        new_sol = chosen_op["function"](prob, incumbent)
        op_stats[chosen_op["name"]]["counter"] += 1
        
        feasibility, _ = feasibility_check(new_sol, prob)
        if feasibility:
            # print(f"Chosen operator: {chosen_op['name']}")
            new_cost = cost_function(new_sol, prob)
            delta_E = new_cost - incumbent_cost
            
            # Finding a better solution than the best solution
            if new_cost < best_cost:
                best_sol = new_sol.copy()
                best_cost = new_cost
                i_since_best = 0
                incumbent = new_sol.copy()
                incumbent_cost = new_cost
                op_stats[chosen_op["name"]]["score"] += 4
                counter += 4
                
            else:
                incumbent, incumbent_cost, score = acceptance_probability(new_sol, incumbent, incumbent_cost, i, total_iterations, best_cost, new_cost, feasibility, delta_E)
                i_since_best += 1
                op_stats[chosen_op["name"]]["score"] += score
                counter += score
            
        # Updating selection parameters for the operators, and iterate i
        if i != 0 and i % 100 == 0:
            # update the weights/probability of the operators, based on the scores
            # if counter > 0:
            #     for op in operators:
            #         op_stats[op["name"]]["probability"] = (op_stats[op["name"]]["score"] / counter)
            #         if i % 1000 == 0:
            #             print(f"Operator: {op['name']}, Probability: {op_stats[op['name']]['probability']}")
            #         # history["operator_probs"][op["name"]].append(op_stats[op["name"]]["probability"])
            #         # history["operator_scores"][op["name"]].append(op_stats[op["name"]]["score"])
            #         op_stats[op["name"]]["score"] = 0
            # counter = 0
            

            # Record scores before updating
            for op in operators:
                op_name = op["name"]
                # history["operator_scores"][op_name].append(op_stats[op_name]["score"])
            
            # Mark segment boundary
            # history["segment_boundaries"].append(current_iteration)
            
            total_score = sum(op_stats[op["name"]]["score"] for op in operators)
            total_count = sum(op_stats[op["name"]]["counter"] for op in operators)
            
            if total_score > 0 and total_count > 0:
                # Calculating new probabilities for each operator
                for op in operators:
                    op_name = op["name"]
                    counter = max(1, op_stats[op_name]["counter"])
                    score = op_stats[op_name]["score"]
                    
                    # Calculating normalized scores
                    op_stats[op_name]["probability"] = (score / counter + 0.01) / (total_score / total_count + 0.03 * len(operators))
                    
                # Normalize probabilities to sum to 1
                prob_sum = sum(op_stats[op["name"]]["probability"] for op in operators)
                if prob_sum > 0:
                    for op in operators:
                        op_stats[op["name"]]["probability"] /= prob_sum
                        if i % 1000 == 0:
                            print(f"Operator: {op['name']}, Probability: {op_stats[op['name']]['probability']}")
                
            
            
            
            # Record updated probabilities
            for op in operators:
                op_name = op["name"]
                # history["operator_probs"][op_name].append(op_stats[op_name]["probability"])
            
            # Reset scores and counters for the next segment
            for op in operators:
                op_stats[op["name"]]["score"] = 0
                op_stats[op["name"]]["counter"] = 0
           
    return best_sol, op_stats


 