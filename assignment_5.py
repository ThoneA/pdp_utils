import math
import random
from pdp_utils.Utils import *


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


# def OP1(prob, sol):
#     # return weighted_removal(prob, sol, soft_greedy_reinsert)
#     # return random_removal_1(prob, sol, soft_greedy_reinsert)
#     return dummy_removal(prob, sol, soft_greedy_reinsert) # 7_calls, 49,88%

# def OP2(prob, sol):
#     return random_removal_2(prob, sol, soft_greedy_reinsert) # 7_calls, 36%

# def OP3(prob, sol):
#     # return weighted_removal(prob, sol, random_reinsert)
#     # return random_removal_1(prob, sol, random_reinsert)
#     # return random_removal_2(prob, sol, random_reinsert)
#     return dummy_removal(prob, sol, random_reinsert)

# def OP4(prob, sol): 
#     return weighted_removal(prob, sol, easy_reinsert)# 18_calls has a probability of 73,33%
#     # return random_removal_1(prob, sol, easy_reinsert)
#     # return random_removal_2(prob, sol, easy_reinsert)
#     # return dummy_removal(prob, sol, easy_reinsert)

# def OP5(prob, sol):
#     return weighted_removal(prob, sol, easy_shuffle_reinsert) # 7_calls, has a probability of 66,15%
#     # return random_removal_1(prob, sol, easy_shuffle_reinsert)
#     # return random_removal_2(prob, sol, easy_shuffle_reinsert)

# def OP6(prob, sol):
#     return dummy_removal(prob, sol, easy_shuffle_reinsert) # 7_calls, 36%
    
# def OP7(prob, sol): 
#     # return weighted_removal(prob, sol, empty_reinsert)
#     return random_removal_1(prob, sol, empty_reinsert) # 7_calls, has a probability of 40%, # 35_calls, har a probability of 71,61%

# def OP8(prob, sol):
#     return random_removal_2(prob, sol, empty_reinsert) # 7_calls, 32,9%, 53_calls, 75%

# def OP9(prob, sol):
#     return dummy_removal(prob, sol, empty_reinsert) # 35_calls, 40%

def OP1(prob, sol):
    new_sol =  random_removal_1(prob, sol, soft_greedy_reinsert_2)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = random_removal_1(prob, sol, soft_greedy_reinsert_2)
        
    return new_sol

def OP2(prob, sol):
    new_sol = random_removal_2(prob, sol, soft_greedy_reinsert_2)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = random_removal_2(prob, sol, soft_greedy_reinsert_2)
    
    return new_sol

def OP3(prob, sol):
    new_sol = dummy_removal(prob, sol, soft_greedy_reinsert_2)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = dummy_removal(prob, sol, soft_greedy_reinsert_2)
    return new_sol

def OP4(prob, sol):
    new_sol = weighted_removal(prob, sol, soft_greedy_reinsert_2)
    # for i in range(10):
    #     if new_sol != sol:
    #         break
    #     new_sol = weighted_removal(prob, sol, soft_greedy_reinsert_2)
    return new_sol



# def OP1(prob, sol): 
#      new_sol = sol.copy()
#      vehicle_ranges = zero_pos(sol)
#      biggest_weight = 0
#      calls_to_reinsert = [] 
#      calls = prob['n_calls']
 
#      for vehicle_index, (start, end) in enumerate(vehicle_ranges):      
#          vehicle_calls = new_sol[start:end]
#          unique_calls = set(vehicle_calls)
#          unique_calls.discard(0)
 
#          if not unique_calls:
#              continue
 
#          calls_list = list(unique_calls)
#          call_indices = np.array([x - 1 for x in calls_list if x - 1 < calls])
 
#          if len(call_indices) > 0:
#              vehicle_weight = np.sum(prob['Cargo'][call_indices, 2])
 
#              if vehicle_weight > biggest_weight:
#                  biggest_weight = vehicle_weight
#                  calls_to_reinsert = calls_list
 
#      if calls_to_reinsert:
#         if len(calls_to_reinsert) < 10:
#             num_to_select = np.random.randint(1, len(calls_to_reinsert) + 1)
#         else:
#             num_to_select = np.random.randint(1, 10)

#         selected_calls = np.random.choice(calls_to_reinsert, num_to_select, replace=False)

#         # Remove selected calls
#         new_sol = [x for x in new_sol if x not in selected_calls]
#         # new_sol = easy_reinsert(selected_calls, prob, new_sol)
        
#         x = "1234"
#         chosen_reinsertion = random.choice(x)
        
#         if chosen_reinsertion == "1":
#             new_sol = random_reinsert(selected_calls, prob, new_sol)
#         elif chosen_reinsertion == "2":
#             new_sol = easy_reinsert(selected_calls, prob, new_sol)
#         elif chosen_reinsertion == "3":
#             new_sol = easy_shuffle_reinsert(selected_calls, prob, new_sol)
#         elif chosen_reinsertion == "4":
#             new_sol = soft_greedy_reinsert(selected_calls, prob, new_sol)
    
#         return new_sol

"""
This operator randomly chooses between 2 and 10 calls depending on the size of the file.
Then inserst the calls back into the solution by using a easy_shuffle_reinsert.
# """
# def OP2(prob, sol):
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

#     # new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
#     # new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
#     # new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
#     x = "1234"
#     chosen_reinsertion = random.choice(x)

#     if chosen_reinsertion == "1":
#         new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "2":
#         new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "3":
#         new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "4":
#         new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    
#     # new_sol = reinsert(calls_to_reinsert, prob, new_sol)
        
#     return new_sol

"""
This operator chooses randomly a car that contains calls,
then it randomly chooses calls between 1 and 10.
"""
# def OP3(prob, sol):
#     new_sol = sol.copy()
#     vehicles = prob['n_vehicles']
#     vehicle_ranges = zero_pos(sol)

#     chosen_vehicle_index = np.random.randint(0, vehicles + 1)

#     while vehicle_ranges[chosen_vehicle_index][0] == vehicle_ranges[chosen_vehicle_index][1]:
#         chosen_vehicle_index = np.random.randint(0, vehicles + 1)

#     start, end = vehicle_ranges[chosen_vehicle_index]

#     vehicle_calls = new_sol[start:end]
#     unique_calls = set(vehicle_calls)
#     unique_calls.discard(0)    
#     calls_list = list(unique_calls)

#     if len(calls_list) < 10:
#         calls_n = np.random.randint(1, len(calls_list) + 1)
#     elif len(calls_list) >= 10:
#         calls_n = np.random.randint(2, 10)

#     calls_to_reinsert = []
#     while calls_n > 0:
#         call = np.random.choice(calls_list)
#         calls_list.remove(call)
#         calls_n -= 1
#         new_sol.remove(call)
#         new_sol.remove(call)
#         calls_to_reinsert.append(call)

#     # new_sol = greedy_reinsert(calls_to_reinsert, prob, new_sol)
#     new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
#     # new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
#     x = "1234"
#     chosen_reinsertion = random.choice(x)
    
#     if chosen_reinsertion == "1":
#         new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "2":
#         new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "3":
#         new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "4":
#         new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    
#     # new_sol = reinsert(calls_to_reinsert, prob, new_sol)

#     return new_sol

"""
This operator checks if there are any calls in the dummy that can be inserted into any of the vehicles
"""
# def OP4(prob, sol):
#     new_sol = sol.copy()
#     vehicle_ranges = zero_pos(new_sol)
    
#     vehicle_index, (start, end) = list(enumerate(vehicle_ranges))[-1]
#     vehicle_calls = new_sol[start:end]
#     unique_calls = set(vehicle_calls)
#     unique_calls.discard(0)
#     calls_list = list(unique_calls)
    
#     if end > start:
#         if len(calls_list) < 10:
#             calls_n = np.random.randint(1, len(calls_list) + 1)
#         elif len(calls_list) >= 10:
#             calls_n = np.random.randint(2, 10)
            
#     calls_to_reinsert = random.sample(range(1, len(calls_list) + 1), calls_n)  
    
#     # Remove selected calls
#     new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
#     # new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
#     x = "1234"
#     chosen_reinsertion = random.choice(x)
    
#     if chosen_reinsertion == "1":
#         new_sol = random_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "2":
#         new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "3":
#         new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
#     elif chosen_reinsertion == "4":
#         new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    
#     # new_sol = reinsert(calls_to_reinsert, prob, new_sol)
    
#     return new_sol  

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from pdp_utils.Utils import *

def general_adaptive_metaheuristics(prob, initial_sol, segment_size=100, plot_results=True):
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    T_f = 0.1  
    total_iterations = 9900
    n_calls = prob['n_calls']
    
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
    
    iterations_since_best = 0
    # escape_condition = 1000  # After this many iterations without improvement, apply escape

    incumbent_cost = cost_function(incumbent, prob)
    best_cost = incumbent_cost

    delta_w = []
    
    # For plotting
    history = {
        "iterations": [],
        "best_costs": [],
        "operator_probs": {op["name"]: [] for op in operators},
        "operator_scores": {op["name"]: [] for op in operators},
        "segment_boundaries": []
    }
    
    counter = 0
    
    # Initial warmup phase
    for w in range(1, 100): 
        # Select operator based on current probabilities
        probabilities = [op_stats[op["name"]]["probability"] for op in operators]
        chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
        chosen_op = operators[chosen_op_idx]
        
        # Apply selected operator
        op_stats[chosen_op["name"]]["counter"] += 1
        new_sol = chosen_op["function"](prob, incumbent)
            
        feasibility, _ = feasibility_check(new_sol, prob)
        if feasibility:
            counter += 1
        
        if not feasibility:
            continue
        
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
        
        if feasibility and delta_E < 0:
            incumbent = new_sol.copy()
            incumbent_cost = new_cost
            
            # Better than incumbent: +2 points
            op_stats[chosen_op["name"]]["score"] += 2
            
            if incumbent_cost < best_cost:
                best_sol = incumbent.copy()
                best_cost = incumbent_cost
                iterations_since_best = 0
           
                # New best solution: +4 points
                op_stats[chosen_op["name"]]["score"] += 4
                
                # Record the iteration where we found a better solution
                history["iterations"].append(w)
                history["best_costs"].append(best_cost)
               
        elif feasibility:
            if random.random() < 0.8:
                incumbent = new_sol.copy()
                incumbent_cost = new_cost
            print(f"delta_e: {delta_E}")
            delta_w.append(delta_E)
            
            # Feasible but not better: +1 point
            op_stats[chosen_op["name"]]["score"] += 1
    
    if delta_w:
        delta_avg = np.mean(delta_w) 
        T_0 = -delta_avg / math.log(0.8)
    else:
        T_0 = 1.0  
        
    alpha = (T_f / T_0) ** (1/total_iterations) 
    T = T_0
    
    # Main phase
    for i in range(1, total_iterations):
        current_iteration = 100 + i  # Offset by warm-up phase
        
        if iterations_since_best == 500:
            print(f'No improvement for {iterations_since_best} iterations. Current best cost: {best_cost}, and current cost: {incumbent_cost}')
            
            # accepted_solutions = 0
            
            # while accepted_solutions < 20:
                # probabilities = [op_stats[op["name"]]["probability"] for op in operators]
                # chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
                # op1 = operators['P1']
                
                # op_stats[chosen_op["name"]]["counter"] += 1
                # new_sol = op1["function"](prob, incumbent)
            for i in range(n_calls * 3): # KANSKJE HOPPE LITT LENGRE ENN 20?
                if i == 0:
                    print(f"Trying to escape local optimum...")
                new_sol = OP1(prob, incumbent)
                
                feasibility, _ = feasibility_check(new_sol, prob)
                if not feasibility:
                    continue
                
                new_cost = cost_function(new_sol, prob)
                delta_E = new_cost - incumbent_cost
                
                incumbent = new_sol.copy()
                incumbent_cost = new_cost
                # accepted_solutions += 1
                
                if new_cost < best_cost:
                    best_sol = new_sol.copy()
                    best_cost = new_cost
                    iterations_since_best = 0
                    print(f"Found a better solution: {best_cost} at iteration {current_iteration}")
                    break
        
            iterations_since_best = 0
        # Checking if we need to escape local optimum
        # if iterations_since_best > escape_condition:
        #     vehicle_ranges = zero_pos(incumbent)
            
        #     for vehicle_index, (start, end) in enumerate(vehicle_ranges):
        #         if end > start:
        #             segment = list(incumbent[start:end])
        #             if len(segment) > 2:
        #                 random.shuffle(segment)
        #                 incumbent[start:end] = segment
        #             break
                
        #     iterations_since_best = 0
        #     incumbent_cost = cost_function(incumbent, prob)
            
        # Updating operator probabilities periodically
        if i % segment_size == 0 and i > 0:
            # Record scores before updating
            for op in operators:
                op_name = op["name"]
                history["operator_scores"][op_name].append(op_stats[op_name]["score"])
            
            # Mark segment boundary
            history["segment_boundaries"].append(current_iteration)
            
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
            
            # Record updated probabilities
            for op in operators:
                op_name = op["name"]
                history["operator_probs"][op_name].append(op_stats[op_name]["probability"])
            
            # Reset scores and counters for the next segment
            for op in operators:
                op_stats[op["name"]]["score"] = 0
                op_stats[op["name"]]["counter"] = 0
            
        if i % 1000 == 0:
            print(f"Iteration {current_iteration}, Best Cost: {best_cost}")
            
        # Select and apply operator
        probabilities = [op_stats[op["name"]]["probability"] for op in operators]
        chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
        chosen_op = operators[chosen_op_idx]
        
        op_stats[chosen_op["name"]]["counter"] += 1
        new_sol = chosen_op["function"](prob, incumbent)

        feasibility, _ = feasibility_check(new_sol, prob)
        if feasibility:
            counter += 1
            # print(f"counter: {counter}")
        if not feasibility:
            continue
        
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
        # print(f"incumbent: {incumbent}, new: {incumbent_cost}, delta_e: {delta_E}")
        # print(f"new_sol: {new_sol}, new: {new_cost}, delta_e: {delta_E}")
        # print(f"delta_e: {delta_E}")
        
        g = i
        G = total_iterations
        D = 0.2 * ((G-g)/G) * best_cost
        max_acceptable_cost = best_cost + D
        
        if feasibility and delta_E < 0:
            incumbent = new_sol.copy()
            incumbent_cost = new_cost
            
            # Better than incumbent: +2 points
            op_stats[chosen_op["name"]]["score"] += 2
            
            if incumbent_cost < best_cost:
                best_sol = incumbent.copy()
                best_cost = incumbent_cost
                iterations_since_best = 0
                
                # New best solution: +4 points
                op_stats[chosen_op["name"]]["score"] += 4
                
                # Record the iteration where we found a better solution
                history["iterations"].append(current_iteration)
                history["best_costs"].append(best_cost)
            else:
                iterations_since_best += 1
                
        # elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
        elif feasibility and (new_cost <= max_acceptable_cost):
            incumbent = new_sol.copy()
            incumbent_cost = new_cost
            iterations_since_best += 1
            
            # Feasible but not better: +1 point
            op_stats[chosen_op["name"]]["score"] += 1
                
        else:
            iterations_since_best += 1

        T = alpha * T
    
    # # Create plots if requested
    # if plot_results:
    #     plot_optimization_history(history)
    
   
    
    return best_sol, op_stats, history


def plot_combined_optimization_history(histories, file_name):
    """
    Create plots combining histories from multiple runs.
    
    Parameters:
    - histories: List of history dictionaries from different runs
    - file_name: Name of the problem file for the plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Best solution cost over iterations for all runs
    plt.subplot(2, 1, 1)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(histories)))
    
    for i, history in enumerate(histories):
        plt.plot(history["iterations"], history["best_costs"], 
                 color=colors[i], alpha=0.7, label=f"Run {i+1}")
        
    plt.title(f"Best Solution Cost Over Iterations for {file_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Average operator probabilities over segments
    plt.subplot(2, 1, 2)
    
    # Calculate average probabilities across all runs
    all_op_names = set()
    for history in histories:
        all_op_names.update(history["operator_probs"].keys())
    
    # Find the maximum segment length
    max_segments = max([len(next(iter(h["operator_probs"].values()))) for h in histories if h["operator_probs"]])
    segment_points = list(range(max_segments))
    
    # For each operator, calculate the average probability over all runs
    avg_probabilities = {op_name: [] for op_name in all_op_names}
    
    for segment_idx in range(max_segments):
        for op_name in all_op_names:
            segment_values = []
            for history in histories:
                if op_name in history["operator_probs"] and segment_idx < len(history["operator_probs"][op_name]):
                    segment_values.append(history["operator_probs"][op_name][segment_idx])
            
            if segment_values:
                avg_probabilities[op_name].append(sum(segment_values) / len(segment_values))
            else:
                # If no data for this segment, use previous value or 0
                prev_value = avg_probabilities[op_name][-1] if avg_probabilities[op_name] else 0
                avg_probabilities[op_name].append(prev_value)
    
    # Plot average probabilities
    for op_name, probs in avg_probabilities.items():
        if probs:  # Check if we have data
            plt.plot(segment_points[:len(probs)], probs, label=f"Avg {op_name}")
    
    plt.title(f"Average Operator Probabilities Over Segments for {file_name}")
    plt.xlabel("Segment")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"optimization_history_{file_name.replace('.txt', '')}.png")
    
    # Create a separate plot for average operator scores
    plt.figure(figsize=(15, 6))
    
    # Calculate average scores across all runs
    avg_scores = {op_name: [] for op_name in all_op_names}
    
    for segment_idx in range(max_segments):
        for op_name in all_op_names:
            segment_values = []
            for history in histories:
                if op_name in history["operator_scores"] and segment_idx < len(history["operator_scores"][op_name]):
                    segment_values.append(history["operator_scores"][op_name][segment_idx])
            
            if segment_values:
                avg_scores[op_name].append(sum(segment_values) / len(segment_values))
            else:
                # If no data for this segment, use previous value or 0
                prev_value = avg_scores[op_name][-1] if avg_scores[op_name] else 0
                avg_scores[op_name].append(prev_value)
    
    # Plot average scores
    for op_name, scores in avg_scores.items():
        if scores:  # Check if we have data
            plt.plot(segment_points[:len(scores)], scores, label=f"Avg {op_name}")
    
    plt.title(f"Average Operator Scores Over Segments for {file_name}")
    plt.xlabel("Segment")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"operator_scores_{file_name.replace('.txt', '')}.png")
    
    plt.show()


# def plot_optimization_history(history):
#     """
#     Create plots from the optimization history.
#     """
#     plt.figure(figsize=(15, 10))
    
#     # Plot 1: Best solution cost over iterations
#     plt.subplot(2, 1, 1)
#     plt.plot(history["iterations"], history["best_costs"], 'b-')
#     plt.scatter(history["iterations"], history["best_costs"], c='r', s=30)
    
#     # Add vertical lines for segment boundaries
#     for segment in history["segment_boundaries"]:
#         plt.axvline(x=segment, color='gray', linestyle='--', alpha=0.7)
    
#     plt.title("Best Solution Cost Over Iterations")
#     plt.xlabel("Iteration")
#     plt.ylabel("Best Cost")
#     plt.grid(True)
    
#     # Plot 2: Operator probabilities over segments
#     plt.subplot(2, 1, 2)
    
#     # One line per operator
#     segment_points = list(range(0, len(history["segment_boundaries"])))
    
#     for op_name in history["operator_probs"]:
#         if history["operator_probs"][op_name]:  # Check if we have data
#             plt.plot(segment_points, history["operator_probs"][op_name], label=op_name)
    
#     plt.title("Operator Probabilities Over Segments")
#     plt.xlabel("Segment")
#     plt.ylabel("Probability")
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig("optimization_history.png")
    
#     # Create a separate plot for operator scores
#     plt.figure(figsize=(15, 6))
#     segment_points = list(range(0, len(history["segment_boundaries"])))
    
#     for op_name in history["operator_scores"]:
#         if history["operator_scores"][op_name]:  # Check if we have data
#             plt.plot(segment_points, history["operator_scores"][op_name], label=op_name)
    
#     plt.title("Operator Scores Over Segments")
#     plt.xlabel("Segment")
#     plt.ylabel("Score")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("operator_scores.png")
    
#     plt.show()
