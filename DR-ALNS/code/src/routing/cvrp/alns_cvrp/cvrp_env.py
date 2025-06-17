import sys
sys.path.insert(0, 'C:/Users/10133/Desktop/DR-ALNS-master/DR-ALNS/code/src')  # 替换为你的本地项目路径

from routing.cvrp.alns_cvrp import cvrp_helper_functions


def evaluate_solution(routes, tasks_info, distance_matrix, tank_capacity, now_energy,
                      fuel_consumption_rate, charging_rate, velocity):
    total_cost = 0

    for route in routes:
        dist_cost = cvrp_helper_functions.calculate_route_distance(route, distance_matrix)
        # print(dist_cost)
        arrival_times = cvrp_helper_functions.calculate_arrival_times(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                                                fuel_consumption_rate, charging_rate, velocity)
        time_cost = 0
        for i in range(1, len(route)):
            # print(route[i])
            if tasks_info[route[i]]['Type'] is 'f':
                time_cost += 0
            elif tasks_info[route[i]]['Type'] is 'c':
                time_cost += tasks_info[route[i]]['Due-Date']-arrival_times[i-1]
            # print(time_cost)
        route_cost = dist_cost + 0.1 * time_cost + 1000
        total_cost += route_cost

    # can be removed in deployment, just for testing
    # for route in routes:
    #     if compute_route_load(route, demands_data) > truck_capacity:
    #         print('TOO MUCH LOAD FOR TRUCK')
    return total_cost


class cvrpEnv:

    def __init__(self, initial_solution, nb_customers, truck_capacity, dist_matrix_data, dist_depot_data, demands_data, problem_instance, seed):
        self.nb_customers = nb_customers
        self.truck_capacity = truck_capacity
        self.dist_matrix_data = dist_matrix_data
        self.dist_depot_data = dist_depot_data
        self.demands_data = demands_data

        self.seed = seed
        self.problem_instance = problem_instance

        self.routes = initial_solution

    def objective(self, best=False):
        score = evaluate_solution(self.routes, self.truck_capacity, self.dist_matrix_data, self.dist_depot_data, self.demands_data)
        return score
