import sys
sys.path.insert(0, 'C:/Users/10133/Desktop/DR-ALNS-master/DR-ALNS/code/src')
from routing.cvrp.alns_cvrp import cvrp_helper_functions
import random

# --- init solution ---
def compute_initial_solution(current, random_state):
    cust = [c[0] for c in current.customers]
    current.routes = cvrp_helper_functions.innh(cust, current.tasks_info, current.distance_matrix,
                                       current.tank_capacity, current.now_energy,current.fuel_consumption_rate,
                                       current.charging_rate, current.velocity, current.load_capacity)
    current = cvrp_helper_functions.process_route(current, current.nodes, current.tasks_info, current.distance_matrix,
                                       current.tank_capacity, current.now_energy,current.fuel_consumption_rate,
                                       current.charging_rate, current.velocity, current.load_capacity)
    # print(current.routes)
    # routes = []
    # route = []
    # unvisited_customers = [i for i in range(1, current.customers + 1)]
    # # unvisited_customers = [i[0] for i in current.customers]
    # while len(unvisited_customers) != 0:
    #     if len(route) == 0:
    #         random_customer = random.choice(unvisited_customers)
    #         route.append(random_customer)
    #         unvisited_customers.remove(random_customer)
    #     else:
    #         route_load = cvrp_helper_functions.compute_route_load(route, current.demands_data)
    #         unvisited_eligible_customers = cvrp_helper_functions.get_customers_that_can_be_added_to_route(route_load, current.truck_capacity,
    #                                                                                unvisited_customers, current.demands_data)
    #         if len(unvisited_eligible_customers) == 0:
    #             routes.append(route)
    #             route = []  # new_route
    #             random_customer = random.choice(unvisited_customers)
    #             route.append(random_customer)
    #             unvisited_customers.remove(random_customer)
    #         else:
    #             closest_unvisited_customer = cvrp_helper_functions.get_closest_customer_to_add(route, unvisited_eligible_customers,
    #                                                                      current.dist_matrix_data, current.dist_depot_data)
    #             route.append(closest_unvisited_customer)
    #             unvisited_customers.remove(closest_unvisited_customer)
    #
    # if route != []:
    #     routes.append(route)
    #
    # current.routes = routes
    # current.graph = cvrp_helper_functions.NeighborGraph(current.nb_customers)

    return current