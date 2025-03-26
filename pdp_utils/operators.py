import random
import numpy as np
from pdp_utils.Utils import *

def zero_pos(sol):
    # Convert to NumPy for more efficient processing
    sol_array = np.array(sol)
    zero_indices = np.where(sol_array == 0)[0]
    vehicle_ranges = []
    start_index = 0  
    
    # Defines vehicle ranges in the solution
    for zero in zero_indices:
        vehicle_ranges.append((start_index, zero)) 
        start_index = zero + 1
    vehicle_ranges.append((start_index, len(sol) - 1))
    
    return vehicle_ranges

def greedy_reinsert(calls, prob, removed_sol):
    best_sol = removed_sol.copy()
    
    for call in calls:
        vehicle_ranges = zero_pos(removed_sol)
        new_best_sol = best_sol.copy()
        new_best_cost = 1e12
        
        for vehicle_index, (start, end) in enumerate(vehicle_ranges):
            # Use NumPy to check vessel cargo more efficiently
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


def soft_greedy_reinsert(calls, prob, removed_sol, temperature=1.0):
    best_sol = removed_sol[:]
    
    for call in calls:
        vehicle_ranges = zero_pos(removed_sol)
        new_best_sol = None
        new_best_cost = float('inf')

        for vehicle_index, (start, end) in enumerate(vehicle_ranges):
            if vehicle_index == prob['n_vehicles']:
                continue
            if prob['VesselCargo'][vehicle_index][call - 1] == 0:
                continue
            
            # Only check a subset of positions instead of all
            pickup_positions = range(start, end + 1)
            delivery_positions = range(start + 1, end + 2)
            
            for p_pos in pickup_positions:
                temp_p_sol = best_sol[:]
                temp_p_sol.insert(p_pos, call)
                
                for d_pos in delivery_positions:
                    temp_d_sol = temp_p_sol[:]
                    temp_d_sol.insert(d_pos, call)
                    
                    if not feasibility_check(temp_d_sol, prob)[0]:
                        continue
                    
                    temp_cost = cost_function(temp_d_sol, prob)
                    
                    if temp_cost < new_best_cost or random.uniform(0, 1) < np.exp(-(temp_cost - new_best_cost) / temperature):
                        new_best_sol = temp_d_sol
                        new_best_cost = temp_cost
            
        if new_best_sol is None:
            best_sol.append(call)
            best_sol.append(call)
        else:
            best_sol = new_best_sol

    return best_sol

"""
This operator chooses the vehicle with the biggest weight, and then chooses
a random number of calls between two and twenty of the calls inside that vehicle. 
"""
def OP1(prob, sol): # Change this operator such that it doesnt calculate all the calls and vehicle weights
    new_sol = sol.copy()
    vehicle_ranges = zero_pos(sol)
    biggest_weight = 0
    calls_to_reinsert = [] 
    
    for vehicle_index, (start, end) in enumerate(vehicle_ranges):      
        vehicle_calls = new_sol[start:end]
        unique_calls = set(vehicle_calls)
        unique_calls.discard(0)
        
        if not unique_calls:
            continue
        
        # Use NumPy for more efficient indexing and calculations
        calls_list = list(unique_calls)
        call_indices = np.array([x - 1 for x in calls_list if x - 1 < prob['n_calls']])
        
        if len(call_indices) > 0:
            # Use NumPy sum for calculating vehicle weight
            vehicle_weight = np.sum(prob['Cargo'][call_indices, 2])
            
            if vehicle_weight > biggest_weight:
                biggest_weight = vehicle_weight
                calls_to_reinsert = calls_list
    
    if calls_to_reinsert:
        if len(calls_to_reinsert) < 20:
            num_to_select = np.random.randint(2, len(calls_to_reinsert) + 1)
        else:
            num_to_select = np.random.randint(2, 20)
        
        selected_calls = np.random.choice(calls_to_reinsert, num_to_select, replace=False)
        
        # Remove selected calls
        new_sol = [x for x in new_sol if x not in selected_calls]
        
        new_sol = soft_greedy_reinsert(selected_calls, prob, new_sol)
                
    return new_sol

"""
This operation randomly chooses between 2 and 10 calls depending on the size of the file.
Then inserst the calls back into the solution by using a soft greedy function.
"""
def OP2(prob, sol):
    new_sol = sol.copy()
    calls = prob['n_calls']
    calls_to_reinsert = []

    # Choose a random number between 2 and 20
    if calls < 10:
        calls_n = np.random.randint(2, calls)
    else:
        calls_n = np.random.randint(2, 10)
    
    calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)

    # Remove selected calls
    new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
    new_sol = soft_greedy_reinsert(calls_to_reinsert, prob, new_sol)
    
    return new_sol



"""
This operator chooses randomly a car that contains calls,
then it randomly chooses calls between 2 and 10.
"""
def OP3(prob, sol):
    new_sol = sol.copy()
    vehicles = prob['n_vehicles']
    vehicle_ranges = zero_pos(sol)
    
    chosen_vehicle_index = np.random.randint(0, vehicles + 1)

    while vehicle_ranges[chosen_vehicle_index][0] == vehicle_ranges[chosen_vehicle_index][1]:
        chosen_vehicle_index = np.random.randint(0, vehicles + 1)
    
    start, end = vehicle_ranges[chosen_vehicle_index]
    
    # Choose unique calls from the vehicle
    vehicle_calls = new_sol[start:end]
    unique_calls = set(vehicle_calls)
    unique_calls.discard(0)    
    calls_list = list(unique_calls)
    
    # Generating a random number of calls to remove
    if len(calls_list) < 10:
        calls_n = np.random.randint(2, len(calls_list) + 1)
    else: 
        calls_n = np.random.randint(2, 10)
    
    calls_to_reinsert = []
    while calls_n > 0:
        call = np.random.choice(calls_list)
        calls_list.remove(call)
        calls_n -= 1
        new_sol.remove(call)
        new_sol.remove(call)
        calls_to_reinsert.append(call)
    
    new_sol = greedy_reinsert(calls_to_reinsert, prob, new_sol)
    
    return new_sol

def upgraded_simulated_annealing(prob, initial_sol):
    """
    Implements the Simulated Annealing algorithm for solving the Pickup and Delivery Problem.
    
    Parameters:
    prob (dict): Problem instance containing problem parameters
    initial_sol (list): Initial solution
    
    Returns:
    list: Best solution found
    """
    # Initialize parameters
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    T_f = 0.1  # Final temperature
    
    # probabilities for operators
    P1, P2, P3 = 1/3, 1/3, 1/3
    operators = ["P1", "P2", "P3"]
    probabilities = [P1, P2, P3]

        
    # Initial cost calculations
    incumbent_cost = cost_function(incumbent, prob)
    best_cost = incumbent_cost
    
    # Tracking temperature and cost changes
    delta_w = []
    
    # First phase: Exploration and delta_w calculation
    for w in range(1, 101):  # Increased range for more thorough exploration
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            new_sol = OP3(prob, incumbent)
            
        feasibility, _ = feasibility_check(new_sol, prob)
        
        if feasibility:
            new_cost = cost_function(new_sol, prob)
            delta_E = new_cost - incumbent_cost
            if delta_E < 0:  # Always accept improvements
                incumbent = new_sol
                incumbent_cost = new_cost
                
                if incumbent_cost < best_cost:
                    best_sol = incumbent
                    best_cost = incumbent_cost
            else:
                # Probabilistic acceptance of worse solutions
                if random.random() < 0.8:
                    incumbent = new_sol
                    incumbent_cost = new_cost
                
                delta_w.append(delta_E)
    
    # Calculate initial temperature
    delta_avg = np.mean(delta_w) 
    T_0 = -delta_avg / math.log(0.8)
    
    # Compute cooling rate
    alpha = (T_f / T_0) ** (1/9900)
    T = T_0
    
    # Main simulated annealing loop
    for i in range(1, 9901):
        chosen_operator = random.choices(operators, weights=probabilities, k=1)[0]
        if chosen_operator == 'P1':
            new_sol = OP1(prob, incumbent)
        elif chosen_operator == 'P2':
            new_sol = OP2(prob, incumbent)
        elif chosen_operator == 'P3':
            new_sol = OP3(prob, incumbent)
            
        feasibility, _ = feasibility_check(new_sol, prob)
       
        if feasibility:
            new_cost = cost_function(new_sol, prob)
            delta_E = new_cost - incumbent_cost
            if delta_E < 0:  # Always accept improvements
                incumbent = new_sol
                incumbent_cost = new_cost
                
                if incumbent_cost < best_cost:
                    best_sol = incumbent
                    best_cost = incumbent_cost
            else:
                # Probabilistic acceptance based on temperature
                acceptance_prob = math.exp(-delta_E / T)
                if random.random() < acceptance_prob:
                    incumbent = new_sol
                    incumbent_cost = new_cost
        
        # Cooling schedule
        T = alpha * T
    
    return best_sol
    


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