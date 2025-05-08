# UiB INF273-Meta-Heuristikkar Pickup and Delivery Problem utils
This repo, has been reconstructed from the original.

# If you get MemoryError try this:
Change to the branch: if crash
This branch does not have caching, which is probably the main reason for the MemoryError :/

# If it crashes try this:
Only run one and one file. 

# If it runs to slow:
Try to remove OP5 in the escape_algorithm().




# Requirement
- numpy

## Usage
### 1) You can install the library by typing the following in terminal:
```bash
python setup.py install
```
### 2) You can load the problem instance using the following line

```bash
from pdp_utils import load_problem

prob = load_problem(*PROBLEM_FILE_ADDRESS*)
```
The function returns dictionary that includes the following information about each problem instance:
- **'n_nodes'**: _number of nodes_
- **'n_vehicles':** _number of vehicles_ 
- **'n_calls':** _number of calls_ 
- **'Cargo':** _information about each call_
- **'TravelTime'**: _for each vehicle the travel time from one node to another_
- **'FirstTravelTime'**: _for each vehicle the travel time from starting point to each node_ 
- **'VesselCapacity'**: _the capacity of each vehicle_
- **'LoadingTime'**: _for each vehicle the loading time of pickup of each call (-1 indicates not allowed)_ 
- **'UnloadingTime'**: _for each vehicle the un-loading time of pickup of each call (-1 indicates not allowed)_ 
- **'VesselCargo'**: _the list of allowed calls for each vehicle_
- **'TravelCost'**: _for each vehicle the travel cost from one node to another_
- **'FirstTravelCost'**:  _for each vehicle the travel cost from starting point to each node_
- **'PortCost'**: _the cost of answering a call for each vehicle (-2 indicates not allowed)_
### 3) You can check the feasiblity of your asnwer like below:
```bash
from pdp_utils import feasibility_check

feasible, log = feasibility_check(SOL, prob)

print(log)
```
### 4) You can check the cost function of a sulotion as below:
```bash
from pdp_utils import cost_function

cost = cost_function(SOL, prob)
```