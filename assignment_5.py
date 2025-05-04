import math
import random
from pdp_utils.Utils import *
import matplotlib.pyplot as plt
import numpy as np

def vehicle_feasibility_check(vehicle_plan, vehicle_index, problem):
    """
    Check if a specific vehicle's plan is feasible
    
    :param vehicle_plan: List of calls (without zeros) assigned to this vehicle
    :param vehicle_index: Index of the vehicle to check
    :param problem: The pickup and delivery problem object
    :return: tuple (is_feasible, reason)
    """
    # Extract necessary problem parameters
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    
    # Convert plan to 0-indexed calls
    vehicle_plan = [call - 1 for call in vehicle_plan if call != 0]
    NoDoubleCallOnVehicle = len(vehicle_plan)
    
    # If the plan is empty, it's trivially feasible
    if NoDoubleCallOnVehicle == 0:
        return True, 'Feasible'
    
    # Check if all calls can be handled by this vehicle
    if not np.all(VesselCargo[vehicle_index, vehicle_plan]):
        return False, 'Incompatible vessel and cargo'
    
    # Calculate load changes throughout the route
    LoadSize = 0
    currentTime = 0
    sortRout = np.sort(vehicle_plan, kind='mergesort')
    I = np.argsort(vehicle_plan, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')
    LoadSize -= Cargo[sortRout, 2]
    LoadSize[::2] = Cargo[sortRout[::2], 2]
    LoadSize = LoadSize[Indx]
    
    # Check if capacity is exceeded at any point
    if np.any(VesselCapacity[vehicle_index] - np.cumsum(LoadSize) < 0):
        return False, 'Capacity exceeded'
    
    # Set up time windows
    Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
    Timewindows[0] = Cargo[sortRout, 6]
    Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
    Timewindows[1] = Cargo[sortRout, 7]
    Timewindows[1, ::2] = Cargo[sortRout[::2], 5]
    Timewindows = Timewindows[:, Indx]
    
    # Get port indices
    PortIndex = Cargo[sortRout, 1].astype(int)
    PortIndex[::2] = Cargo[sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1
    
    # Calculate loading/unloading times
    LU_Time = UnloadingTime[vehicle_index, sortRout]
    LU_Time[::2] = LoadingTime[vehicle_index, sortRout[::2]]
    LU_Time = LU_Time[Indx]
    
    # Calculate travel times
    if NoDoubleCallOnVehicle > 1:
        Diag = TravelTime[vehicle_index, PortIndex[:-1], PortIndex[1:]]
        FirstVisitTime = FirstTravelTime[vehicle_index, int(Cargo[vehicle_plan[0], 0] - 1)]
        RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))
    else:
        # Only one call
        FirstVisitTime = FirstTravelTime[vehicle_index, int(Cargo[vehicle_plan[0], 0] - 1)]
        RouteTravelTime = np.array([FirstVisitTime])
    
    # Check time window constraints
    ArriveTime = np.zeros(NoDoubleCallOnVehicle)
    for j in range(NoDoubleCallOnVehicle):
        ArriveTime[j] = max(currentTime + RouteTravelTime[j], Timewindows[0, j])
        if ArriveTime[j] > Timewindows[1, j]:
            return False, f'Time window exceeded at call {vehicle_plan[j] + 1}'
        currentTime = ArriveTime[j] + LU_Time[j]
    
    return True, 'Feasible'

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
def soft_greedy_reinsert_2(calls, prob, removed_sol, feasibility_cache, cost_cache):
    best_sol = removed_sol
    vehicle_ranges = zero_pos(removed_sol)
    vehicles_n = prob['n_vehicles']
    for call in calls:
        possible_vehicles = np.where(prob['VesselCargo'][:, call - 1] == 1)[0]
        if len(possible_vehicles) >= 3:
            selected_vehicle = np.random.choice(possible_vehicles, size=3, replace=False)
        else:
            selected_vehicle = np.random.choice(possible_vehicles, replace=False)
          
        new_best_sol = best_sol.copy()
        new_best_cost = 1e12
        if isinstance(selected_vehicle, np.int64):
            selected_vehicle = [selected_vehicle]
        else:
            selected_vehicle = list(selected_vehicle)
        
        for vehicle in selected_vehicle:
            start, end = list(vehicle_ranges)[vehicle]
            
        # PICKUP
            for p_pos in range(start, end + 1):
                # DELIVERY
                for d_pos in range(p_pos + 1, end + 2):
                    temp_sol = best_sol.copy()
                    temp_sol.insert(p_pos, call)
                    temp_sol.insert(d_pos, call)
                    
                    sol_key = tuple(temp_sol)
                    
                    if sol_key in feasibility_cache:
                        feasibility, _ = feasibility_cache[sol_key]
                    else:
                        new_vehicle_ranges = zero_pos(temp_sol)
                        v_start = new_vehicle_ranges[vehicle][0]
                        v_end = new_vehicle_ranges[vehicle][1]
                        vehicle_plan = temp_sol[v_start:v_end]
                        feasibility, _ = vehicle_feasibility_check(vehicle_plan, selected_vehicle, prob)
                        feasibility_cache[sol_key] = (feasibility, _)
                    
                    if feasibility:
                        if sol_key in cost_cache:
                            temp_cost = cost_cache[sol_key]
                        else:
                            temp_cost = cost_function(temp_sol, prob)
                            cost_cache[sol_key] = temp_cost
                    
                    if temp_cost < new_best_cost:
                        new_best_sol = temp_sol
                        new_best_cost = temp_cost
            
        if call in new_best_sol:
            best_sol = new_best_sol.copy()

        # Dummy reinsertion
        else:
            best_sol.insert(len(best_sol), call)
            best_sol.insert(len(best_sol), call)           

    return best_sol   
        

"""
This reinsertion function checks the two best positions for the pickup and delivery of the calls, then it checks the difference between the cost of the two solution for every call, and then chooses the one that has the biggest difference.
"""            
def k_regret(calls, prob, removed_sol, feasibility_cache, cost_cache):
    best_sol = removed_sol
    vehicles_n = prob['n_vehicles']
    remaining_calls = set(calls)
    score = 0
        
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
                    for d_pos in range(p_pos + 1, end + 2):
                        temp_sol = best_sol.copy()
                        temp_sol.insert(p_pos, call)
                        temp_sol.insert(d_pos, call)
                        
                        sol_key = tuple(temp_sol)
                        
                        if sol_key in feasibility_cache:
                            feasibility, _ = feasibility_cache[sol_key]
                        else:
                            new_vehicle_ranges = zero_pos(temp_sol)
                            v_start= new_vehicle_ranges[vehicle_index][0]
                            v_end = new_vehicle_ranges[vehicle_index][1]
                            vehicle_plan = temp_sol[v_start:v_end]
                            feasibility, _ = vehicle_feasibility_check(vehicle_plan, vehicle_index, prob)
                            feasibility_cache[sol_key] = (feasibility, _)
                            # score += 1
                        
                        if feasibility:
                            if sol_key in cost_cache:
                                temp_cost = cost_cache[sol_key]
                            else:
                                temp_cost = cost_function(temp_sol, prob)
                                cost_cache[sol_key] = temp_cost
                                
                            k_best_positions.append((temp_sol, temp_cost))
                            # print(f"k_best_positions: {k_best_positions}")
            
            # Keep only the k best positions
            if len(calls) >= 20:
                k = 4
            elif 20 > len(calls) >= 10:
                k = 3
            else:
                k = 2
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
                for d_pos in range(p_pos + 1, end + 2):
                    temp_sol = best_sol.copy()
                    temp_sol.insert(p_pos, biggest_diff_call)
                    temp_sol.insert(d_pos, biggest_diff_call)
                    
                    sol_key = tuple(temp_sol)
                        
                    if sol_key in feasibility_cache:
                        feasibility, _ = feasibility_cache[sol_key]
                    else:
                        new_vehicle_ranges = zero_pos(temp_sol)
                        v_start= new_vehicle_ranges[vehicle_index][0]
                        v_end = new_vehicle_ranges[vehicle_index][1]
                        vehicle_plan = temp_sol[v_start:v_end]
                        feasibility, _ = vehicle_feasibility_check(vehicle_plan, vehicle_index, prob)
                        feasibility_cache[sol_key] = (feasibility, _)
                        # score += 1
                    
                    if feasibility:
                        if sol_key in cost_cache:
                            temp_cost = cost_cache[sol_key]
                        else:
                            temp_cost = cost_function(temp_sol, prob)
                            cost_cache[sol_key] = temp_cost
                                
                        if temp_cost < new_best_cost:
                            new_best_sol = temp_sol
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
    
    best_cost = cost_function(best_sol, prob)
    
    return best_sol, best_cost, score
        
        
     
"""
This operator chooses the vehicle with the biggest weight, and then chooses
a random number of calls between one and ten of the calls inside that vehicle. 
# """
def weighted_removal(prob, sol, reinsert, feasibility_cache, cost_cache): 
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
            num_to_select = np.random.randint(1, len(calls_to_reinsert) + 1 )
        else:
            num_to_select = np.random.randint(2, int(len(calls_to_reinsert) * 0.6))
        
        selected_calls = np.random.choice(calls_to_reinsert, num_to_select, replace=False)
        
        # Remove selected calls
        new_sol = [x for x in new_sol if x not in selected_calls]
        
        new_sol, new_cost, score = reinsert(selected_calls, prob, new_sol, feasibility_cache, cost_cache)
      
    return new_sol, new_cost, score

"""
This operator randomly chooses between 1 and 10 calls depending on the size of the file.
Then inserst the calls back into the solution by using a easy_shuffle_reinsert.
"""
def random_removal_1(prob, sol, reinsert, feasibility_cache, cost_cache, boolean):
    new_sol = sol.copy()
    calls = prob['n_calls']
    calls_to_reinsert = []

    if boolean == True:
        if calls >= 25:
            calls_n = np.random.randint(2, 25)
        else:
            calls_n = np.random.randint(1, calls + 1)
    else:
        if calls < 10:
            calls_n = np.random.randint(1, calls + 1)
        else:
            calls_n = np.random.randint(2, 10)
    
    calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)
    
    # Remove selected calls
    new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
    new_sol, new_cost, score = reinsert(calls_to_reinsert, prob, new_sol, feasibility_cache, cost_cache)
        
    return new_sol, new_cost, score

"""
This operator chooses randomly a car that contains calls,
then it randomly chooses calls between 1 and 10.
"""
def random_removal_2(prob, sol, reinsert, feasibility_cache, cost_cache):
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
    
    new_sol, new_cost, score = reinsert(calls_to_reinsert, prob, new_sol, feasibility_cache, cost_cache)
    
    return new_sol, new_cost, score

"""
This operator checks if there are any calls in the dummy that can be inserted into any of the vehicles
"""
def dummy_removal(prob, sol, reinsert, feasibility_cache, cost_cache):
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
        else:
            calls_n = np.random.randint(2,10)
    
    calls_to_reinsert = random.sample(range(1, len(calls_list) + 1), calls_n)
    
    # Remove selected calls
    new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
    new_sol, new_cost, score = reinsert(calls_to_reinsert, prob, new_sol, feasibility_cache, cost_cache)
    
    return new_sol, new_cost, score


def OP1(prob, sol, feasibility_cache, cost_cache):
    new_sol, new_cost, score =  random_removal_1(prob, sol, k_regret, feasibility_cache, cost_cache, False)

    return new_sol, new_cost, score

def OP2(prob, sol, feasibility_cache, cost_cache):
    new_sol, new_cost, score = random_removal_2(prob, sol, k_regret, feasibility_cache, cost_cache)

    return new_sol, new_cost, score

def OP3(prob, sol, feasibility_cache, cost_cache):
    new_sol, new_cost, score = dummy_removal(prob, sol, k_regret, feasibility_cache, cost_cache)

    return new_sol, new_cost, score

def OP4(prob, sol, feasibility_cache, cost_cache):
    new_sol, new_cost, score = weighted_removal(prob, sol, k_regret, feasibility_cache, cost_cache)

    return new_sol, new_cost, score

def OP5(prob, sol, feasibility_cache, cost_cache):
    new_sol, new_cost, score =  random_removal_1(prob, sol,  k_regret, feasibility_cache, cost_cache, True)

    return new_sol, new_cost, score


def escape_algorithm(prob, incumbent, incumbent_cost, best_sol, best_cost, i_since_best, feasibility_cache, cost_cache):
    for i in range(20):
        if i == 0:
            print(f'No improvement for {i_since_best} iterations. Current best cost: {best_cost}, and current cost: {incumbent_cost}')
            print(f"Trying to escape local optimum...")
        
        if prob['n_calls'] >= 25:
            new_sol, new_cost, score = OP5(prob, incumbent, feasibility_cache, cost_cache)
        else:
            new_sol, new_cost, score = OP1(prob, incumbent, feasibility_cache, cost_cache)

        delta_E = new_cost - incumbent_cost
        
        incumbent = new_sol.copy()
        incumbent_cost = new_cost
        
        if new_cost < best_cost:
            best_sol = new_sol.copy()
            best_cost = new_cost
            i_since_best = 0
            print(f"Found a better solution: {best_cost} at iteration {i}\nBest Sol: {best_sol}")
            break

    return incumbent, incumbent_cost, best_sol, best_cost, i_since_best, score
   
def acceptance_probability(new_sol, incumbent, incumbent_cost, i, total_iterations, best_cost, new_cost, delta_E):
    g = i
    G = total_iterations
    D = 0.2 * ((G-g)/G) * best_cost
    max_acceptable_cost = best_cost + D
    score = 0
    
    # If I find a better solution than the current solution
    if delta_E < 0:
        incumbent = new_sol.copy()
        # incumbent_sol = new_sol.copy()
        incumbent_cost = new_cost
        score += 2
        
    elif new_cost <= max_acceptable_cost:
        incumbent = new_sol.copy()
        incumbent_cost = new_cost
        # score += 1 # SKAL JEG HA DENNE????

    return incumbent, incumbent_cost, score
        
     

def general_adaptive_metaheuristics(prob, initial_sol, plot_results = True): 
    feasibility_cache = {}
    cost_cache = {}
    
    best_sol = initial_sol.copy()
    incumbent = initial_sol.copy()
    
    sol_key = tuple(incumbent)
    incumbent_cost = cost_function(incumbent, prob)
    cost_cache[sol_key] = incumbent_cost
    best_cost = incumbent_cost
    
    i_since_best = 0
    total_iterations = 10000
    counter = 0

    operators = [
        {"name": "P1", "function": OP1},
        {"name": "P2", "function": OP2},
        {"name": "P3", "function": OP3},
        {"name": "P4", "function": OP4},
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
        "incumbent_costs": [],
        "operator_probs": {op["name"]: [] for op in operators},
        "operator_usage": {op["name"]: [] for op in operators},
        "best_found_at": [],  # Track iterations when best solutions are found
        "segment_boundaries": []
    }
    
    
    # Hvilke stop condition skal jeg ha?
    for i in range(total_iterations):
        # print(f"Iteration {i}, Best Cost: {best_cost}")
        if i % 1000 == 0:
            print(f"********Iteration {i}, Best Cost: {best_cost}, \nBest Sol: {best_sol}********") 
        
        if i_since_best != 0 and i_since_best % 500 == 0:
            incumbent, incumbent_cost, best_sol, best_cost, i_since_best, new_score = escape_algorithm(prob, incumbent, incumbent_cost, best_sol, best_cost, i_since_best, feasibility_cache, cost_cache)
        
        # new_sol = incumbent.copy()
        
        # Choose a operator depending on selection parameters to apply to new_sol
        probabilities = [op_stats[op["name"]]["probability"] for op in operators]
        chosen_op_idx = random.choices(range(len(operators)), weights=probabilities, k=1)[0]
        chosen_op = operators[chosen_op_idx]
        
        # Apply selected operator
        new_sol, new_cost, new_score = chosen_op["function"](prob, incumbent, feasibility_cache, cost_cache)
        op_stats[chosen_op["name"]]["counter"] += 1
        if new_score != 0:
            op_stats[chosen_op["name"]]["score"] += new_score

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
            history["best_found_at"].append(i)
            print(f"found a better solution at iteration {i}, with cost {best_cost}\nBest Sol: {best_sol}")
        else:
            incumbent, incumbent_cost, score = acceptance_probability(new_sol, incumbent, incumbent_cost, i, total_iterations, best_cost, new_cost, delta_E)
            i_since_best += 1
            op_stats[chosen_op["name"]]["score"] += score
            counter += score
            
        # Updating selection parameters for the operators, and iterate i
        if i != 0 and i % 100 == 0:
            # Record scores before updating
            for op in operators:
                op_name = op["name"]
                # history["operator_scores"][op_name].append(op_stats[op_name]["score"])
            
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
                
            
            history["iterations"].append(i)
            history["best_costs"].append(best_cost)
            history["incumbent_costs"].append(incumbent_cost)
            
            # Record updated probabilities
            for op in operators:
                op_name = op["name"]
                # history["operator_probs"][op_name].append(op_stats[op_name]["probability"])
                history["operator_probs"][op_name].append(op_stats[op_name]["probability"])
                # Track operator usage count
                history["operator_usage"][op_name].append(op_stats[op_name]["counter"])
            
            # Reset scores and counters for the next segment
            for op in operators:
                op_stats[op["name"]]["score"] = 0
                op_stats[op["name"]]["counter"] = 0
            

           

                

           
    return best_sol, op_stats, history



def plot_optimization_history(history, filename, save_fig=False):
    """Plot the optimization progress"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Objective function over iterations
    plt.subplot(2, 2, 1)
    plt.plot(history["iterations"], history["best_costs"], 'b-', label='Best Cost')
    plt.plot(history["iterations"], history["incumbent_costs"], 'r-', alpha=0.5, label='Incumbent Cost')
    
    # Mark when best solutions were found
    if history["best_found_at"]:
        best_costs = []
        for i in history["best_found_at"]:
            # Find closest recorded iteration
            closest_idx = np.argmin(np.abs(np.array(history["iterations"]) - i))
            if closest_idx < len(history["best_costs"]):
                best_costs.append(history["best_costs"][closest_idx])
            else:
                best_costs.append(history["best_costs"][-1])
        
        plt.scatter(history["best_found_at"], best_costs, c='green', marker='*', s=100, label='New Best Found')
    
    plt.title(f'Optimization Progress for {filename}')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Operator Probabilities
    plt.subplot(2, 2, 2)
    for op_name, probs in history["operator_probs"].items():
        plt.plot(history["iterations"], probs, label=op_name)
    
    plt.title('Operator Probabilities Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Operator Usage
    plt.subplot(2, 2, 3)
    for op_name, usage in history["operator_usage"].items():
        plt.plot(history["iterations"], usage, label=op_name)
    
    plt.title('Operator Usage Count')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Improvement Rate
    plt.subplot(2, 2, 4)
    
    # Calculate percentage improvement from initial cost
    if history["best_costs"]:
        initial_cost = history["best_costs"][0]
        improvement = [(initial_cost - cost) / initial_cost * 100 for cost in history["best_costs"]]
        plt.plot(history["iterations"], improvement, 'g-')
        plt.title('Improvement Rate (%)')
        plt.xlabel('Iteration')
        plt.ylabel('Improvement (%)')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'optimization_history_{filename}.png', dpi=300)
    plt.show()

def plot_convergence_statistics(all_histories, filename, save_fig=False):
    """Plot statistics across multiple runs"""
    if not all_histories:
        return
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Best costs across all runs
    plt.subplot(2, 2, 1)
    for i, history in enumerate(all_histories):
        plt.plot(history["iterations"], history["best_costs"], alpha=0.3, label=f'Run {i+1}')
    
    # Calculate and plot average
    avg_iterations = all_histories[0]["iterations"]  # Assume all have same iterations
    avg_costs = np.zeros_like(avg_iterations, dtype=float)
    
    for history in all_histories:
        avg_costs += np.array(history["best_costs"])
    
    avg_costs /= len(all_histories)
    plt.plot(avg_iterations, avg_costs, 'k-', linewidth=2, label='Average')
    
    plt.title(f'Convergence Across Runs for {filename}')
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: When best solutions were found
    plt.subplot(2, 2, 2)
    all_best_found = []
    for history in all_histories:
        all_best_found.extend(history["best_found_at"])
    
    plt.hist(all_best_found, bins=20, alpha=0.7)
    plt.title('Distribution of When Best Solutions Were Found')
    plt.xlabel('Iteration')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot 3: Average operator probabilities
    plt.subplot(2, 2, 3)
    op_names = list(all_histories[0]["operator_probs"].keys())
    
    for op_name in op_names:
        avg_probs = np.zeros_like(avg_iterations, dtype=float)
        for history in all_histories:
            avg_probs += np.array(history["operator_probs"][op_name])
        avg_probs /= len(all_histories)
        
        plt.plot(avg_iterations, avg_probs, label=op_name)
    
    plt.title('Average Operator Probabilities')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Final performance boxplot
    plt.subplot(2, 2, 4)
    final_costs = [history["best_costs"][-1] for history in all_histories]
    plt.boxplot(final_costs)
    plt.title('Final Solution Quality')
    plt.ylabel('Cost')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(f'convergence_statistics_{filename}.png', dpi=300)
    plt.show()
 