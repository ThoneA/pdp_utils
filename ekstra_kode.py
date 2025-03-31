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
                    inserted_calls = True
                    break
        
        if not inserted_calls:
            new_sol.append(call)
            new_sol.append(call)
    
    return new_sol



def OP4(prob, sol):
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
            
        calls_to_reinsert = random.sample(calls_list, calls_n)  
    # calls_to_reinsert = random.sample(range(1, calls + 1), calls_n)
    
    # Remove selected calls
        new_sol = [x for x in new_sol if x not in calls_to_reinsert]
    
    # new_sol = easy_shuffle_reinsert(calls_to_reinsert, prob, new_sol)
        new_sol = empty_reinsert(calls_to_reinsert, prob, new_sol)
        return new_sol
    
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
                        # np.random.shuffle(segment)
                        # incumbent[start:end] = segment
                        segment_list = list(segment)
                        random.shuffle(segment_list) 
                        incumbent[start:end] = segment_list
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
    
            
        
    
