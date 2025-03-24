# Can I only focus on the removal of calls? And creating different ways to remove calls? 
# And then reinsert the calls in the same way for all the operators.
# Therefore I can create a function that reinserts the calls, that the operators can use. 

import random

from pdp_utils.Utils import *


# Helping method: This method returns the vehicle ranges.
def zero_pos(sol):
    zero_pos = [i for i, x in enumerate(sol) if x == 0]
    vehicle_ranges = []
    start_index = 0  
    
    # Defines vehicle ranges in the solution
    for zero in zero_pos:
        vehicle_ranges.append((start_index, zero)) 
        start_index = zero + 1
    vehicle_ranges.append((start_index, len(sol) - 1))
    
    return vehicle_ranges

# Helping method: This method adds the removed calls back into the solution, in a Greedy way.
def greedy_reinsert(calls, sol, prob, removed_sol):
    # Greedy reinsertion of calls,
    # Put the call in the best possible position in the solution.
    # The best possible position is the position that minimizes the cost.
    best_sol = removed_sol.copy() # initial solution
  
    
    for call in calls:
        vehicle_ranges = zero_pos(removed_sol)
        new_best_sol = best_sol.copy()
        new_best_cost = 1e12
          # PICKUP
        
        # print(f'New best sol: {new_best_sol}')
        # print(f'Vehicle ranges: {vehicle_ranges}')
        
        for vehicle_index, (start, end) in enumerate(vehicle_ranges):
            # SJEKK OM CALL KAN I VEHICLE INDEX, HVIS IKKE CONTINUE
            if vehicle_index == prob['n_vehicles']:
                continue
            if  prob['VesselCargo'][vehicle_index][call - 1] == 0:
                # print(f'Vehicle {vehicle_index + 1} cant take call {call}: {prob["VesselCargo"][vehicle_index]}')
                continue
            
            # print(f'Vehicle index: {vehicle_index}')
            
          
            for p_pos in range(start, end + 1):
                # if p_pos == start: # BURDE JEG SJEKKE HER OM DENNE LØSNINGEN ER FEASIBLE FØR JEG SIER DEN ER BESTE?
                #     new_best_sol.insert(p_pos, call) # pickuo
                #     new_best_sol.insert(p_pos + 1, call) # delivery
                #     new_best_cost = cost_function(new_best_sol, prob)
                #     continue
                    
                temp_p_sol = best_sol.copy() # Henter ut beste løsning med forrige calls
                temp_p_sol.insert(p_pos, call)
                
                # DELIVERY             
                for d_pos in range(p_pos + 1, end + 2):
                    temp_d_sol = temp_p_sol.copy()
                    temp_d_sol.insert(d_pos, call)
                    
                
                    feasibility, _ = feasibility_check(temp_d_sol, prob)
                    # print("Feasibility -", feasibility)
                    if feasibility:
                        temp_cost = cost_function(temp_d_sol, prob)
                        # print(f'Feasible solution: {temp_d_sol}')
                        
                        if temp_cost < new_best_cost:
                            # print("New best cost", temp_cost)
                            new_best_sol = temp_d_sol
                            new_best_cost = temp_cost
                            
        if new_best_cost == 1e12:
            # print("No improvemnets found")
            best_sol.insert(len(best_sol), call) # pickup
            best_sol.insert(len(best_sol), call) # delivery
        else:
            best_sol = new_best_sol
        
    # print(best_sol)
    return best_sol
        
   
# Her kan jeg endre slik at den ser etter vehicle med størst vekt
def OP1(prob, sol):
    # Remove multiple calls, given a criteria.
    # The criteria is that a car is full/not capable to take more calls, which is a reason to remove calls.
    # So if a car is full, then remove a random given number of calls and greedy reinsert them.
    
    # If no cars are full, then remove a random given number of calls from the dummy route.
    new_sol = sol.copy()
    for i in range(100):
        print(i)
        vehicles = prob['n_vehicles']
        vehicle_ranges = zero_pos(sol)
        
        # possible_calls_to_reinsert = []
        chosen_vehicle_index = 0
        biggest_weight = 0
        calls_to_reinsert = []
        
        for vehicle_index, (start, end) in enumerate(vehicle_ranges):      
            # calls = set(new_sol[start:end])
            # calls = list(calls)
            # calls = [x - 1 for x in calls]
            vehicle_calls = new_sol[start:end]
            unique_calls = set(vehicle_calls)
            unique_calls.discard(0)
            
            if not unique_calls:
                continue
            
            calls_list = list(unique_calls)
            call_indices = [x - 1 for x in calls_list if x - 1 < prob['n_calls']]
            
            if call_indices:
                vehicle_weight = sum(prob['Cargo'][call_indices, 2]) 
                
                if vehicle_weight > biggest_weight:
                        chosen_vehicle_index = vehicle_index 
                        biggest_weight = vehicle_weight
                        calls_to_reinsert = calls_list
            
        if calls_to_reinsert:
            # Choosing random number of calls to remove from the possible_calls_to_reinsert
            num_to_select = random.randint(1, len(calls_to_reinsert))
            selected_calls = random.sample(calls_to_reinsert, num_to_select)
            # print(f'Selected calls: {selected_calls}')
            # Removing the calls from the new solution
            new_sol = [x for x in new_sol if x not in selected_calls]
            
            new_sol = greedy_reinsert(selected_calls, sol, prob, new_sol)
            
            
                
    return new_sol
    
    
    

def OP2():
    # Remove multiple calls, given a criteria.
    # Criteria: the car that is the fullest. Altså: has the smallest remaining capacity.
    
    return
    

def OP3():
    # Remove multiple calls, given a criteria. 
    # Criterie: Randomly choose a car and remove a random number of calls from that car.
    return 



# # Her kan jeg endre slik at den ser etter største vekt under ett gitt call
# def OP1(prob, sol):
#     # Remove multiple calls, given a criteria.
#     # The criteria is that a car is full/not capable to take more calls, which is a reason to remove calls.
#     # So if a car is full, then remove a random given number of calls and greedy reinsert them.
    
#     # If no cars are full, then remove a random given number of calls from the dummy route.
#     new_sol = sol.copy()
#     vessel_capacity = prob['VesselCapacity']
#     vehicles = prob['n_vehicles']
#     vehicle_ranges = zero_pos(sol)
    
#     possible_calls_to_reinsert = []
#     calls_to_reinsert = []
#     for vehicle_index, (start, end) in enumerate(vehicle_ranges):
#         if vehicle_index == vehicles: # This is the dummy route
#             # posible_calls_to_reinsert += sol[start:end]
#             # chosen_vehicle_index = vehicle_index
#             calls_to_reinsert = [] # TODO - Velge et tilfeldig antall calls å fjerne ifra dummy
#         else:
#             calls = set(sol[start:end])
#             calls = list(calls)
#             calls = [x - 1 for x in calls]
            
#             capacity = vessel_capacity[vehicle_index]
#             vehicle_weight = sum(prob['Cargo'][calls, 2])
            
#             calls_for_vehicle = prob['VesselCargo'][vehicle_index]
#             calls_for_vehicle = [i + 1 for i, x in enumerate(calls_for_vehicle) if x == 1]
#             print(calls_for_vehicle)
            
#             calls_not_in_vehicle = [x for x in calls_for_vehicle if x not in calls]
#             print(calls_not_in_vehicle)
            
#             dict = {}
#             for call in calls_not_in_vehicle:
#                 dict[call] = prob['Cargo'][call - 1, 2]
                        
#             # Finne minste mulig call for gitt bil, som ikke er i bilen
#             smallest_possible_call = min(dict, key=dict.get)
#             # Må kanskje hentes ut på en annen måte,  Må jeg plusse på 1 her for å få riktig bil?
#             if capacity - vehicle_weight < dict[smallest_possible_call]:
#                 chosen_vehicle = vehicle_index 
            
        
#         calls_to_reinsert = chosen_vehicle # velg random antall av callsene til chosen vehicle 
                
#     return 
    
    