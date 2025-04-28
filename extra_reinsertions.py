
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
   