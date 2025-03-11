import math
import random
import numpy as np
from collections import namedtuple


def load_problem(filename):
    """

    :rtype: object
    :param filename: The address to the problem input file
    :return: named tuple object of the problem attributes
    """
    A = []
    B = [] # B inneholder hvilke calls som skal til hvilken bil
    C = []
    D = []
    E = []
    with open(filename) as f:
        lines = f.readlines()
        num_nodes = int(lines[1])
        num_vehicles = int(lines[3])
        num_calls = int(lines[num_vehicles + 5 + 1])

        for i in range(num_vehicles):
            A.append(lines[1 + 4 + i].split(','))

        for i in range(num_vehicles):
            B.append(lines[1 + 7 + num_vehicles + i].split(','))

        for i in range(num_calls):
            C.append(lines[1 + 8 + num_vehicles * 2 + i].split(','))

        for j in range(num_nodes * num_nodes * num_vehicles):
            D.append(lines[1 + 2 * num_vehicles + num_calls + 9 + j].split(','))

        for i in range(num_vehicles * num_calls):
            E.append(lines[1 + 1 + 2 * num_vehicles + num_calls + 10 + j + i].split(','))
        f.close()

    Cargo = np.array(C, dtype=np.double)[:, 1:]
    D = np.array(D, dtype=int)

    TravelTime = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    TravelCost = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    for j in range(len(D)):
        TravelTime[D[j, 0]][D[j, 1], D[j, 2]] = D[j, 3]
        TravelCost[D[j, 0]][D[j, 1], D[j, 2]] = D[j, 4]

    VesselCapacity = np.zeros(num_vehicles)
    StartingTime = np.zeros(num_vehicles)
    FirstTravelTime = np.zeros((num_vehicles, num_nodes))
    FirstTravelCost = np.zeros((num_vehicles, num_nodes))
    A = np.array(A, dtype=int)
    for i in range(num_vehicles):
        VesselCapacity[i] = A[i, 3]
        StartingTime[i] = A[i, 2]
        for j in range(num_nodes):
            FirstTravelTime[i, j] = TravelTime[i + 1, A[i, 1], j + 1] + A[i, 2]
            FirstTravelCost[i, j] = TravelCost[i + 1, A[i, 1], j + 1]
    TravelTime = TravelTime[1:, 1:, 1:]
    TravelCost = TravelCost[1:, 1:, 1:]
    VesselCargo = np.zeros((num_vehicles, num_calls + 1))
    B = np.array(B, dtype=object)
    for i in range(num_vehicles):
        VesselCargo[i, np.array(B[i][1:], dtype=int)] = 1
    VesselCargo = VesselCargo[:, 1:]

    LoadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    UnloadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    PortCost = np.zeros((num_vehicles + 1, num_calls + 1))
    E = np.array(E, dtype=int)
    for i in range(num_vehicles * num_calls):
        LoadingTime[E[i, 0], E[i, 1]] = E[i, 2]
        UnloadingTime[E[i, 0], E[i, 1]] = E[i, 4]
        PortCost[E[i, 0], E[i, 1]] = E[i, 5] + E[i, 3]

    LoadingTime = LoadingTime[1:, 1:]
    UnloadingTime = UnloadingTime[1:, 1:]
    PortCost = PortCost[1:, 1:]
    output = {
        'n_nodes': num_nodes,
        'n_vehicles': num_vehicles,
        'n_calls': num_calls,
        'Cargo': Cargo,
        'TravelTime': TravelTime,
        'FirstTravelTime': FirstTravelTime,
        'VesselCapacity': VesselCapacity,
        'LoadingTime': LoadingTime,
        'UnloadingTime': UnloadingTime,
        'VesselCargo': VesselCargo,
        'TravelCost': TravelCost,
        'FirstTravelCost': FirstTravelCost,
        'PortCost': PortCost
    }
    return output


def feasibility_check(solution, problem):
    """

    :rtype: tuple
    :param solution: The input solution of order of calls for each vehicle to the problem
    :param problem: The pickup and delivery problem object
    :return: whether the problem is feasible and the reason for probable infeasibility
    """
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    solution = np.append(solution, [0])
    ZeroIndex = np.array(np.where(solution == 0)[0], dtype=int)
    feasibility = True
    tempidx = 0
    c = 'Feasible'
    for i in range(num_vehicles):
        currentVPlan = solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1
        if NoDoubleCallOnVehicle > 0:

            if not np.all(VesselCargo[i, currentVPlan]):
                feasibility = False
                c = 'incompatible vessel and cargo'
                break
            else:
                LoadSize = 0
                currentTime = 0
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')
                LoadSize -= Cargo[sortRout, 2]
                LoadSize[::2] = Cargo[sortRout[::2], 2]
                LoadSize = LoadSize[Indx]
                if np.any(VesselCapacity[i] - np.cumsum(LoadSize) < 0):
                    feasibility = False
                    c = 'Capacity exceeded'
                    break
                Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
                Timewindows[0] = Cargo[sortRout, 6]
                Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
                Timewindows[1] = Cargo[sortRout, 7]
                Timewindows[1, ::2] = Cargo[sortRout[::2], 5]

                Timewindows = Timewindows[:, Indx]

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                LU_Time = UnloadingTime[i, sortRout]
                LU_Time[::2] = LoadingTime[i, sortRout[::2]]
                LU_Time = LU_Time[Indx]
                Diag = TravelTime[i, PortIndex[:-1], PortIndex[1:]]
                FirstVisitTime = FirstTravelTime[i, int(Cargo[currentVPlan[0], 0] - 1)]

                RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

                ArriveTime = np.zeros(NoDoubleCallOnVehicle)
                for j in range(NoDoubleCallOnVehicle):
                    ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
                    if ArriveTime[j] > Timewindows[1, j]:
                        feasibility = False
                        c = 'Time window exceeded at call {}'.format(j)
                        break
                    currentTime = ArriveTime[j] + LU_Time[j]

    return feasibility, c


def cost_function(Solution, problem):
    """

    :param Solution: the proposed solution for the order of calls in each vehicle
    :param problem:
    :return:
    """

    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']


    NotTransportCost = 0
    RouteTravelCost = np.zeros(num_vehicles)
    CostInPorts = np.zeros(num_vehicles)

    Solution = np.append(Solution, [0])
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    tempidx = 0

    for i in range(num_vehicles + 1):
        currentVPlan = Solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1

        if i == num_vehicles:
            NotTransportCost = np.sum(Cargo[currentVPlan, 3]) / 2
        else:
            if NoDoubleCallOnVehicle > 0:
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                Diag = TravelCost[i, PortIndex[:-1], PortIndex[1:]]

                FirstVisitCost = FirstTravelCost[i, int(Cargo[currentVPlan[0], 0] - 1)]
                RouteTravelCost[i] = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
                CostInPorts[i] = np.sum(PortCost[i, currentVPlan]) / 2

    TotalCost = NotTransportCost + sum(RouteTravelCost) + sum(CostInPorts)
    return TotalCost

def initial_solution(problem):
    num_vehicles = problem['n_vehicles']
    sol = [0] * num_vehicles
    for i in range(problem['n_calls']):
        sol.append(i + 1)
        sol.append(i + 1)
    return sol

def random_function(problem):
    num_vehicles = problem['n_vehicles']
    vessel_cargo = problem['VesselCargo']
    print(vessel_cargo)
    cargo_volume = problem['Cargo'][:, 2]
    vessel_capacity = problem['VesselCapacity']
    
    final_route = []
    dummy_route = []
    assigned_calls = set()
    not_assigned_calls = set()
    
    # INITIAL SOLUTION
    initial_sol = [0] * num_vehicles #[0,0,0,1,2,3,4,5,6,7]
    for i in range(problem['n_calls']):
        initial_sol.append(i+1)
    
    for i in range(num_vehicles):
        vehicle_route = []
        calls_for_vehicle = np.where(vessel_cargo[i] == 1)[0] + 1 # JEg må addere med 1 for å få riktig call
        calls_for_vehicle = calls_for_vehicle[calls_for_vehicle != 0]
        np.random.shuffle(calls_for_vehicle)
       
        current_load = 0
        
        # PICKUP
        for call in calls_for_vehicle:
            if call not in assigned_calls and current_load + cargo_volume[call-1] <= vessel_capacity[i]:
                vehicle_route.append(call)
                current_load += cargo_volume[call-1]
                assigned_calls.add(call)
                not_assigned_calls.discard(call)
            else:
                not_assigned_calls.add(call)
       
        # DELIVERY
        for call in vehicle_route: 
            if vehicle_route.count(call) == 1:
                insert_pos = np.random.randint(vehicle_route.index(call) + 1, len(vehicle_route) + 1)
                vehicle_route.insert(insert_pos, call)
        final_route.append(vehicle_route)
      
    result = []
    route_index = 0
    
    for item in initial_sol:
        if item == 0 and route_index < len(final_route):
            result.extend(final_route[route_index])
            route_index += 1
            result.append(item)
        if item not in result:
            dummy_route.append(item)
        
    # The DUMMY ROUTE
    for call in dummy_route:
        if dummy_route.count(call) == 1:
            insert_pos = np.random.randint(dummy_route.index(call) + 1, len(dummy_route) + 1)
            dummy_route.insert(insert_pos, call)
    result.extend(dummy_route)
    
    return result

# this should remove one call from the initial solution (both pickup and delivery)
# then add this call to one of the vehicles
def n_operator(prob, sol):
    new_sol = sol.copy()
    
    # Pick a random non-zero call
    possible_calls = [x for x in new_sol if x != 0]
    if not possible_calls:
        return new_sol
    
    call = np.random.choice(possible_calls)
    
    # Count zeroes and find which vehicle has our call
    zero_positions = [i for i, x in enumerate(new_sol) if x == 0]
    vehicle_ranges = []
    
    # Create ranges for each vehicle
    start_idx = 0
    for zero_pos in zero_positions:
        vehicle_ranges.append((start_idx, zero_pos))
        start_idx = zero_pos + 1
    vehicle_ranges.append((start_idx, len(new_sol)))
    
    # Find which vehicle has our call
    current_vehicle = -1
    for i, (start, end) in enumerate(vehicle_ranges):
        if call in new_sol[start:end]:
            current_vehicle = i
            break
    
    # Remove the call from the solution
    new_sol = [x for x in new_sol if x != call]
    
    # Recalculate cehicle ranges after removal
    zero_positions = [i for i, x in enumerate(new_sol) if x == 0]
    vehicle_ranges = []
    
    start_idx = 0
    for zero_pos in zero_positions:
        vehicle_ranges.append((start_idx, zero_pos))
        start_idx = zero_pos + 1
    vehicle_ranges.append((start_idx, len(new_sol)))
    
    # Choose a different vehicle
    target_vehicle = current_vehicle
    while target_vehicle == current_vehicle and len(vehicle_ranges) > 1:
        target_vehicle = np.random.randint(0, len(vehicle_ranges))
    
    # Get target vehicle range
    start, end = vehicle_ranges[target_vehicle]
    
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
    
    
    
    # Find which vehicle has our call
    
    # call_choice = sol.copy()
    # np.random.shuffle(call_choice)

    # # Here I choose which call to remove and reinsert
    # call = None
    # for i in range(len(call_choice)):
    #     if call_choice[i] != 0:
    #         call = [call_choice[i]] * 2
    #         break

    # # If no call is chosen
    # if not call:
    #     return new_sol
    
    # # Finds out how many zeroes there are before the initial call
    # zero_pos_counter = 0
    # for i in range(len(new_sol)): 
    #     if new_sol[i] == 0:
    #         zero_pos_counter += 1
    #     if new_sol[i] == call[0]:
    #         break

       
    
    # old_sol = new_sol
    # #This removes all instances of the call from the list
    # new_sol = list(filter((call[0]).__ne__, new_sol)) 

    # # fist count how many zeroes there are, then choose a random zero
    # zero_counter = new_sol.count(0)
    
    # # I have to be sure that I don't put the call back into the same vehicle
    # random_zero = np.random.randint(0, zero_counter + 1)
    # while random_zero == zero_pos_counter:
    #     random_zero = np.random.randint(0, zero_counter + 1)
  
    # # INSTANCE 1: RANDOM BECOMES 0
    # if random_zero == 0:
    #     next_zero_pos = None
    #     for i in range(len(new_sol)):
    #         if new_sol[i] == 0:
    #             next_zero_pos = i
    #             if next_zero_pos == 0:
    #                 new_sol.insert(0, call[0])
    #                 new_sol.insert(1, call[1])
    #                 return new_sol
    #             else:
    #                 break
    #     pickup_index = np.random.randint(0, next_zero_pos + 1)
    #     new_sol.insert(pickup_index, call[0])
    #     delivery_index = np.random.randint(pickup_index + 1, next_zero_pos + 2)
    #     new_sol.insert(delivery_index, call[1])
    #     return new_sol
    
    # # INSTANCE 2: REINSERT AFTER THE FIRST 0 
    # zero_count = 1
    # next_zero_index = None
    # for i in range(new_sol.index(0) + 1, len(new_sol)):
    #     if random_zero == zero_count:
    #         for j in range(i, len(new_sol)):
    #             if new_sol[j] == 0:
    #                 next_zero_index = j
    #                 break
    #         if next_zero_index != None:
    #             pickup_index = np.random.randint(i, next_zero_index + 1)
    #             new_sol.insert(pickup_index, call[0])
    #             delivery_index = np.random.randint(pickup_index + 1, next_zero_index + 2)
    #             new_sol.insert(delivery_index, call[1])
    #             break
    #         else:
    #             return old_sol
            
    #     elif new_sol[i] == 0:
    #         zero_count += 1
    
    # return new_sol

def local_search(problem, initial_sol):
    best_sol = initial_sol
    best_cost = cost_function(best_sol, problem)

    
    for i in range(1000):
        new_sol = n_operator(problem, best_sol)
        feasibility, c = feasibility_check(new_sol, problem)
        new_cost = cost_function(new_sol, problem)
        if feasibility and new_cost < best_cost:
            best_sol = new_sol
            best_cost = new_cost
    
    return best_sol

def simulated_annealing(problem, initial_sol):
    best_sol = initial_sol 
    incumbent = initial_sol
    T_f = 0.1
    delta_w = []
    incumbent_cost = cost_function(incumbent, problem)
    best_cost = incumbent_cost
    
    for w in range(1, 100):
        new_sol = n_operator(problem, incumbent)
        feasibility, _ = feasibility_check(new_sol, problem)
        c = cost_function(new_sol, problem)
        delta_E = c - incumbent_cost
        
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = c
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
        elif feasibility:
            if random.random() < 0.8:
                incumbent = new_sol
                incumbent_cost = c
            delta_w.append(delta_E)
        
    delta_avg = np.mean(delta_w)
    print(f"delta_avg: {delta_avg}")
    

        
    T_0 = -delta_avg / math.log(0.8)
    print(f"T_0: {T_0}")
    

    alpha = (T_f / T_0) ** (1/9900)
    T = T_0
    
    for i in range(1, 9900):
        new_sol = n_operator(problem, incumbent)
        c = cost_function(new_sol, problem)
        feasibility, _ = feasibility_check(new_sol, problem)
        delta_E = c - incumbent_cost
        
        if i % 1000 == 0:
            print(f"Vi er her: {i}")
           
        if feasibility and delta_E < 0:
            incumbent = new_sol
            incumbent_cost = c
            if incumbent_cost < best_cost:
                best_sol = incumbent
                best_cost = incumbent_cost
        elif feasibility and (random.random() < (math.exp((-1) * delta_E / T))):
            incumbent = new_sol
            incumbent_cost = c
        
        T = alpha * T
        
    return best_sol
        
    
    