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

"""
This operator chooses the vehicle with the biggest weight, and then chooses
a random number of calls between one and ten of the calls inside that vehicle. 
"""
# GJØR SLIK AT DENNE OPERATOREN VELGER EN RANDOM VEHICLE_INDEX SOM IKKE ER BLITT SJEKKET, BRUK WHILE-LØKKE
# HVIS BILEN HAR OVER ... I VEKT, SÅ SKAL DEN FJERNES CALLS IFRA
def OP1(prob, sol): # Change this operator such that it doesnt calculate all the calls and vehicle weights,
    # but maybe choose one vehicle randomly, and if it is full we will use it!
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
        
        # new_sol = soft_greedy_reinsert(selected_calls, prob, new_sol)
        new_sol = easy_reinsert(selected_calls, prob, new_sol)
                
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


def general_adaptive_metaheuristics(prob, initial_sol, segment_size = 100):
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    T_f = 0.1  
    
    # probabilities for operators
    operators = ["P1", "P2", "P3"]
    P1, P2, P3 = 1/3, 1/3, 1/3
    probabilities = [P1, P2, P3]
    
    # Scores:
    OP1_score = 0
    OP2_score = 0
    OP3_score = 0
    
    # Usage counter:
    OP1_counter = 0
    OP2_counter = 0
    OP3_counter = 0
    
    iterations_since_best = 0
    
    escape_condition = 1000 # After this many iterations without improvement, apply escape

    incumbent_cost = cost_function(incumbent, prob)
    best_cost = incumbent_cost

    delta_w = []
    
    for w in range(1, 100): 
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
    
        if chosen_operator == 'P1':
            OP1_counter += 1
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            OP2_counter += 1
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            OP3_counter += 1
            new_sol = OP3(prob, incumbent)            
            
        feasibility, _ = feasibility_check(new_sol, prob)
        if not feasibility:
            continue
        
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
        
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = new_cost
            
            #  hvis jeg finner en bedre_sol en current_sol gi OP?_score 2 poeng
            if chosen_operator == 'P1':
                OP1_score += 2
            elif chosen_operator == 'P2':
                OP2_score += 2
            elif chosen_operator == 'P3':
                OP3_score += 2            
            
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
                iterations_since_best = 0
                
                # hvis jeg finner en new_best, så gi OP?_score 4 poeng
                if chosen_operator == 'P1':
                    OP1_score += 4
                elif chosen_operator == 'P2':
                    OP2_score += 2
                elif chosen_operator == 'P3':
                    OP3_score += 4
               
        elif feasibility:
            if random.random() < 0.8:
                incumbent = new_sol
                incumbent_cost = new_cost
            delta_w.append(delta_E)
            
            # hvis jeg finner en new_sol som ikke er funnet før, gi OP?_score 1 poeng
            if chosen_operator == 'P1':
                OP1_score += 1
            elif chosen_operator == 'P2':
                OP2_score += 1
            elif chosen_operator == 'P3':
                OP3_score += 1
    
    delta_avg = np.mean(delta_w) 
    T_0 = -delta_avg / math.log(0.8)

    alpha = (T_f / T_0) ** (1/9900) 
    T = T_0
    
    # Her tror jeg at jeg skal oppdatere 'the weights' av operatorene. Altså prosentandelen for at de ulike kjører
    
    for i in range(1, 9900):
        # We have reached local optimum, and have to escape
        if iterations_since_best > escape_condition:
            vehicle_ranges = zero_pos(incumbent)
            
            for vehicle_index, (start, end) in enumerate(vehicle_ranges):
                if end > start:
                    segment = incumbent[start:end]
                    if len(segment) > 2:
                        np.random.shuffle(segment)
                        incumbent[start:end] = segment
                    break
                
            iterations_since_best = 0
            incumbent_cost = cost_function(incumbent, prob)
            
            # Updating the weights according to the scores:
        if i % segment_size == 0 and i > 0: # HVA ER segment_size?
            total_score = OP1_score + OP2_score + OP3_score
            total_count = OP1_counter + OP2_counter + OP3_counter
            if total_score > 0:
                P1 = (OP1_score / max(1, OP1_counter) + 0.01) / (total_score / total_count + 0.03 * len(operators))
                P2 = (OP2_score / max(1, OP2_counter) + 0.01) / (total_score / total_count + 0.03 * len(operators))
                P3 = (OP3_score / max(1, OP3_counter) + 0.01) / (total_score / total_count + 0.03 * len(operators))
            
                weight_sum = P1 + P2 + P3
                P1 /= weight_sum
                P2 /= weight_sum
                P3 /= weight_sum
            
            # Reset scores and counters for the next segment
            OP1_score = 0
            OP2_score = 0
            OP3_score = 0
            
            OP1_counter = 0
            OP2_counter = 0
            OP3_counter = 0
            
            # # Print progress
            # if i % 1000 == 0:
            #     print(f"Iteration {i}, Best Cost: {best_cost}, Current Weights: {weights}")
            
        if i % 1000 == 0:
            print(f"Vi er her: {i}")
            
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            OP1_counter += 1
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            OP2_counter += 1
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            OP3_counter += 1
            new_sol = OP3(prob, incumbent)

        feasibility, _ = feasibility_check(new_sol, prob)
        if not feasibility:
            continue
        
        new_cost = cost_function(new_sol, prob)
        delta_E = new_cost - incumbent_cost
       
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = new_cost
            
            #  hvis jeg finner en bedre_sol enn current_sol gi OP?_score 2 poeng
            if chosen_operator == 'P1':
                OP1_score += 2
            elif chosen_operator == 'P2':
                OP2_score += 2
            elif chosen_operator == 'P3':
                OP3_score += 2   
            
            
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
                
                iterations_since_best = 0
                
                # hvis jeg finner en new_best, så gi OP?_score 4 poeng
                if chosen_operator == 'P1':
                    OP1_score += 4
                elif chosen_operator == 'P2':
                    OP2_score += 2
                elif chosen_operator == 'P3':
                    OP3_score += 4
            
            else:
                iterations_since_best += 1
                
        elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
            incumbent = new_sol
            incumbent_cost = new_cost
            
            iterations_since_best += 1
            
            # hvis jeg finner en new_sol som ikke er funnet før, gi OP?_score 1 poeng
            if chosen_operator == 'P1':
                OP1_score += 1
            elif chosen_operator == 'P2':
                OP2_score += 1
            elif chosen_operator == 'P3':
                OP3_score += 1
                
        else:
            iterations_since_best += 1

        T = alpha * T
    
    return best_sol, best_cost
    
            
        
    
























