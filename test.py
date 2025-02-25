# import numpy as np


# l = [1,2,3,4,0,1,2,0,3,4,5,0]
# x = l.index(0)
# print(x)
# for i in range(x+1, len(l)):
#     print(l[i])
# item = 2
# # print(l)
# l = list(filter((item).__ne__, l))

# print(l)
# call = [7,7]
# zero_count = 3
# rand_zero = 2
# for i in range(len(l)):
#     if l[i] == 0:
#         rand_zero -=1
#         zero_count -=1
#     if rand_zero == 0 and zero_count == 0:
#         pickup_index = np.random.randint(i + 1, len(l) + 1) 
#         l.insert(pickup_index, call[0])
#         delivery_index = np.random.randint(pickup_index + 1, len(l) + 1)
#         l.insert( delivery_index, call[1])
#         break
#     elif rand_zero == 0:
#         next_zero_pos = 0
#         for j in range(i+1, len(l) + 1):
#             if l[j] == 0:
#                 next_zero_pos = j
#                 break
#         print(next_zero_pos)
#         pickup_index = np.random.randint(i + 1, next_zero_pos + 1)
#         l.insert(pickup_index, call[0])
#         delivery_index = np.random.randint(pickup_index + 1, next_zero_pos + 2)
#         l.insert( delivery_index, call[1])
        
#         break
  
# l = [1,2,3]  
# # for i in range(len(l)):
# #     for j in range(i+1, len(l)):
# #         print(j)
    
# print(l)




# call = []
# call = [2] * 2




# l = [1,2,3,4,0,5,6,7,8,0,9,10,11,0,12,13,14]
l = [1,2,0,3]

next_zero_index = None
for i in range(len(l)):
    for j in range(i, len(l)):
        if l[j] == 0:
            next_zero_index = j
            break
    
print(next_zero_index)