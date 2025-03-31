import random
import numpy as np
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

    
def greedy_reinsert(calls, prob, removed_sol): # KANSKJE KUNN SJEKKE HALVPARTEN AV BILENE VELG DEM RANDOM
    best_sol = removed_sol.copy()
    
    for call in calls:
        vehicle_ranges = zero_pos(removed_sol)
        new_best_sol = best_sol.copy()
        new_best_cost = 1e12
        
        for vehicle_index, (start, end) in enumerate(vehicle_ranges):
            if vehicle_index == prob['n_vehicles']:
                continue
            if prob['VesselCargo'][vehicle_index][call - 1] == 0:
                continue
            
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
                            
        if new_best_cost == 1e12:
            best_sol.insert(len(best_sol), call)
            best_sol.insert(len(best_sol), call)
        else:
            best_sol = new_best_sol
        
    return best_sol

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

def soft1_greedy_reinsert(calls, prob, removed_sol):
    best_sol = removed_sol
    vehicle_ranges = zero_pos(removed_sol)
    vehicles_n = prob['n_vehicles']
    
    
    for call in calls:
        selected_vehicle = np.random.randint(0, vehicles_n)
        while prob['VesselCargo'][selected_vehicle][call - 1] == 0:
            selected_vehicle = np.random.randint(0, vehicles_n)
        
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
                # if not feasibility:
                #     continue
                if feasibility:
                    temp_cost = cost_function(temp_d_sol, prob)
                    
                    if temp_cost < new_best_cost:
                        new_best_sol = temp_d_sol
                        new_best_cost = temp_cost
        
        if new_best_cost == 1e12:
            best_sol.insert(len(best_sol), call)
            best_sol.insert(len(best_sol), call)
        
        else:
            best_sol = new_best_sol
    
    return best_sol

"""
This operator chooses the vehicle with the biggest weight, and then chooses
a random number of calls between one and ten of the calls inside that vehicle. 
"""
def OP1(prob, sol): # Change this operator such that it doesnt calculate all the calls and vehicle weights,
    # but maybe choose one vehicle randomly, and if it is full we will use it!
    new_sol = sol.copy()
    for i in range(100):
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
            
        # new_sol = soft_greedy_reinsert(selected_calls, prob, new_sol)
        # new_sol = easy_reinsert(selected_calls, prob, new_sol)
            new_sol = soft1_greedy_reinsert(selected_calls, prob, new_sol)
                
    return new_sol

"""
This operator randomly chooses between 2 and 10 calls depending on the size of the file.
Then inserst the calls back into the solution by using a easy_shuffle_reinsert.
"""
def OP2(prob, sol):
    new_sol = sol.copy()
    calls = prob['n_calls']
    calls_to_reinsert = []

    # Choose a random number between 1 and 10
    if calls < 10:
        calls_n = np.random.randint(1, calls + 1)
    else:
        calls_n = np.random.randint(2, 10)
    
    calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)

    # Remove selected calls
    new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
    # new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
    
    return new_sol



"""
This operator chooses randomly a car that contains calls,
then it randomly chooses calls between 1 and 10.
"""
def OP3(prob, sol):
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
    new_sol = easy_reinsert(calls_to_reinsert, prob, new_sol)
    
    return new_sol

def equal_simulated_annealing(prob, initial_sol):
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    T_f = 0.1  
    
    # probabilities for operators
    P1, P2, P3 = 1/3, 1/3, 1/3
    operators = ["P1", "P2", "P3"]
    probabilities = [P1, P2, P3]

    incumbent_cost = cost_function(incumbent, prob)
    best_cost = incumbent_cost

    delta_w = []
    
    for w in range(1, 100): 
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            new_sol = OP3(prob, incumbent)            
            
        feasibility, c = feasibility_check(new_sol, prob)
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
        
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = new_cost
            
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
        elif feasibility:
            if random.random() < 0.8:
                incumbent = new_sol
                incumbent_cost = new_cost
            delta_w.append(delta_E)
    
    delta_avg = np.mean(delta_w) 
    T_0 = -delta_avg / math.log(0.8)

    alpha = (T_f / T_0) ** (1/9900) 
    T = T_0
    
    for i in range(1, 9900):
        if i % 1000 == 0:
            print(f"Vi er her: {i}")
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            new_sol = OP3(prob, incumbent)

        feasibility, _ = feasibility_check(new_sol, prob)
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
       
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = new_cost
            
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
        elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
            incumbent = new_sol
            incumbent_cost = new_cost

        T = alpha * T
    
    return best_sol
    


def tuned_simulated_annealing(prob, initial_sol):
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    T_f = 0.1 
    
    # Tuned probabilities for operators
    P1, P2, P3 = 0.2, 0.6, 0.2
    operators = ["P1", "P2", "P3"]
    probabilities = [P1, P2, P3]

    incumbent_cost = cost_function(incumbent, prob)
    best_cost = incumbent_cost
 
    delta_w = []
    
    for w in range(1, 100): 
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            new_sol = OP3(prob, incumbent)            
            
        feasibility, c = feasibility_check(new_sol, prob)
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
        
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = new_cost
            
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
        elif feasibility:
            if random.random() < 0.8:
                incumbent = new_sol
                incumbent_cost = new_cost
            delta_w.append(delta_E)

    delta_avg = np.mean(delta_w) 
    T_0 = -delta_avg / math.log(0.8)
    
    alpha = (T_f / T_0) ** (1/9900) 
    T = T_0

    for i in range(1, 9900):
        if i % 1000 == 0:
            print(f"Vi er her: {i}")
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            new_sol = OP3(prob, incumbent)

        feasibility, _ = feasibility_check(new_sol, prob)
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
       
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = new_cost
            
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
        elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
            incumbent = new_sol
            incumbent_cost = new_cost
        
        T = alpha * T
    
    return best_sol



# Different operators I've started to implement, then I've regretted or I am going to try to implement/ideas.

# Criteria: 
# Let the operator pick calls from different vehicles at the same time
# Hva om jeg velger et random stort antall calls også derifra finner 'cost of not transporting' også velger jeg de med høyest verdi.
"""
This operator randomly chooses a quarter of the calls, 
then it sortes the calls depending on their 'cost of not transporting', 
then with a random number between 1 and 20, the operator chooses how many of the calls
will be choosen to reinsert. 
"""
def idiot_OP4(prob, sol):
    new_sol = sol.copy()
    for i in range(1000):
        calls_to_reinsert = []
        calls = prob['n_calls']
        calls_counter = calls        
        quarter_of_calls = []
        
        # Randomly choose a quarter of the calls
        while calls_counter > (calls/4 * 3):
            call_to_select = np.random.randint(1, calls)
            while call_to_select in quarter_of_calls:
                call_to_select = np.random.randint(1, calls)
            quarter_of_calls.append(call_to_select)
            calls_counter -= 1
                
        if quarter_of_calls:
            # Sort quarter_of_calls from the highest value to the lowest by 'cost of not transporting'
            calls_cost_not_t = []
            for i in quarter_of_calls:
                mini = []
                mini.append(i)
                mini.append(prob['Cargo'][i, 3])
                calls_cost_not_t.append(mini)
            sorted_calls = sorted(calls_cost_not_t, key=lambda x: x[1], reverse = True)            
         
            # Then choose a random number between 2 and 20
            if calls < 20:
                num_to_select = np.random.randint(1, calls)
            else: 
                num_to_select = np.random.randint(1, 20)
            
            # Then use this random number to choose the top calls the has the biggest value og 'cost of not transporting'
            calls_to_reinsert = [item[0] for item in sorted_calls[:num_to_select]]
            new_sol = [x for x in new_sol if x not in calls_to_reinsert]
            
            new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
            feasible, _ = feasibility_check(new_sol, prob)
            
            if feasible:
                return new_sol
        
    return sol



"""
This operator chooses randomly a car that contains calls,
then swaps the positions of the calls
"""
def not_OP2(prob, sol):
    new_sol = sol.copy()
    vehicles = prob['n_vehicles']
    vehicle_ranges = zero_pos(sol)
    
    chosen_vehicle_index = np.random.randint(0, vehicles + 1)
    
    
    
    return


# Have an operator that checks if there is any cars in the dumy that can be moved into any of the vehicles

# OP1
# GJØR SLIK AT DENNE OPERATOREN VELGER EN RANDOM VEHICLE_INDEX SOM IKKE ER BLITT SJEKKET, BRUK WHILE-LØKKE
# HVIS BILEN HAR OVER ... I VEKT, SÅ SKAL DEN FJERNES CALLS IFRA
# Change this operator such that it doesnt calculate all the calls and vehicle weights,
# but maybe choose one vehicle randomly, and if it is full we will use it!