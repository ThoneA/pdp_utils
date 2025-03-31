# # import numpy as np


# # l = [1,2,3,4,0,1,2,0,3,4,5,0]
# # x = l.index(0)
# # print(x)
# # for i in range(x+1, len(l)):
# #     print(l[i])
# # item = 2
# # # print(l)
# # l = list(filter((item).__ne__, l))

# # print(l)
# # call = [7,7]
# # zero_count = 3
# # rand_zero = 2
# # for i in range(len(l)):
# #     if l[i] == 0:
# #         rand_zero -=1
# #         zero_count -=1
# #     if rand_zero == 0 and zero_count == 0:
# #         pickup_index = np.random.randint(i + 1, len(l) + 1) 
# #         l.insert(pickup_index, call[0])
# #         delivery_index = np.random.randint(pickup_index + 1, len(l) + 1)
# #         l.insert( delivery_index, call[1])
# #         break
# #     elif rand_zero == 0:
# #         next_zero_pos = 0
# #         for j in range(i+1, len(l) + 1):
# #             if l[j] == 0:
# #                 next_zero_pos = j
# #                 break
# #         print(next_zero_pos)
# #         pickup_index = np.random.randint(i + 1, next_zero_pos + 1)
# #         l.insert(pickup_index, call[0])
# #         delivery_index = np.random.randint(pickup_index + 1, next_zero_pos + 2)
# #         l.insert( delivery_index, call[1])
        
# #         break
  
# # l = [1,2,3]  
# # # for i in range(len(l)):
# # #     for j in range(i+1, len(l)):
# # #         print(j)
    
# # print(l)




# # call = []
# # call = [2] * 2




# # l = [1,2,3,4,0,5,6,7,8,0,9,10,11,0,12,13,14]
# # l = [1,2,0,3]

# # next_zero_index = None
# # for i in range(len(l)):
# #     for j in range(i, len(l)):
# #         if l[j] == 0:
# #             next_zero_index = j
# #             break
    
# # print(next_zero_index)

# from pdp_utils import *
# prob = load_problem('pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt')


# sol = [1,3,2,3,2,1,0,4,4,0,5,6,5,7,6,7,0]

# # calls = sol[0:6]
# calls = set(sol[0:6])
# calls = list(calls)
# calls = [x - 1 for x in calls]

# print(calls) # De som er i bilen
# # Calls bilen kan ta 
# vehicle = 1
# calls_for_vehicle = prob['VesselCargo'][vehicle - 1]
# # calls_for_vehicle er binary, så omgjøre til index+1 hvis 1
# calls_for_vehicle = [i + 1 for i, x in enumerate(calls_for_vehicle) if x == 1]
# print(calls_for_vehicle)

# # Calls som ikke er i bilen
# calls_not_in_vehicle = [x for x in calls_for_vehicle if x not in calls]
# print(calls_not_in_vehicle)
# # Minste size av call som ikke er i bilen
# dict = {}
# for call in calls_not_in_vehicle:
#     dict[call] = prob['Cargo'][call - 1, 2]

# print(dict)

# vehicle_weight = sum(prob['Cargo'][calls, 2])
# print(vehicle_weight)

# capacity = prob['VesselCapacity'][0]

# smallest_possible_call = min(dict, key=dict.get)
# print(smallest_possible_call)


# # Så lenge det ikke er mer plass i bilen til en ny call så skal bilen bli valgt
# if capacity - vehicle_weight < dict[smallest_possible_call]:
#     chosen_vehicle = vehicle
#     print("capacity: " , capacity)
#     print("vehicle_weight: ", vehicle_weight)
#     print("diff: ", capacity - vehicle_weight)
#     print("Chosen vehicle: ", chosen_vehicle)
# # else:
# #     continue

# new_sol = sol.copy()


# possible_calls_to_reinsert = calls_for_vehicle
# print(possible_calls_to_reinsert)
# calls_to_reinsert = []

# new_sol = [x for x in new_sol if x not in calls_to_reinsert]

# num_to_select = random.randint(1, len(possible_calls_to_reinsert))

# calls_to_reinsert = random.sample(possible_calls_to_reinsert, num_to_select)

# new_sol = [x for x in new_sol if x not in calls_to_reinsert]
# new_sol = new_sol + calls_to_reinsert + [8]

# print(calls_to_reinsert)


# def zero_pos(sol):
#     zero_pos = [i for i, x in enumerate(sol) if x == 0]
#     vehicle_ranges = []
#     start_index = 0  
    
#     # Defines vehicle ranges in the solution
#     for zero in zero_pos:
#         vehicle_ranges.append((start_index, zero)) 
#         start_index = zero + 1
#     vehicle_ranges.append((start_index, len(sol) - 1))
    
#     return vehicle_ranges

# print(new_sol)
# vehicle_ranges = zero_pos(new_sol)
# print(vehicle_ranges)

import numpy as np


n = np.random.randint(0, 3)

while n != 3:
    n = np.random.randint(0, 3)

print(n)