import random as rnd
import copy
from routing.cvrp.alns_cvrp import cvrp_helper_functions
import time


# --- regret repair
def get_regret_single_insertion(routes, customer, truck_capacity, distance_matrix_data, distance_depot_data,
                                demands_data):
    # print('python repair')
    insertions = {}
    for route_idx in range(len(routes)):
        if cvrp_helper_functions.compute_route_load(routes[route_idx], demands_data) + demands_data[customer - 1] <= truck_capacity:
            for i in range(len(routes[route_idx]) + 1):
                updated_route = routes[route_idx][:i] + [customer] + routes[route_idx][i:]
                updated_routes = routes[:route_idx] + [updated_route] + routes[route_idx + 1:]
                if i == 0:
                    cost_difference = distance_depot_data[updated_route[0] - 1] + distance_matrix_data[updated_route[0]-1, updated_route[1]-1] - distance_depot_data[updated_route[1]-1]
                elif i == len(routes[route_idx]):
                    cost_difference = distance_depot_data[updated_route[-1] - 1] + distance_matrix_data[updated_route[i-1]-1, updated_route[i]-1] - distance_depot_data[updated_route[i-1]-1]
                else:
                    cost_difference = distance_matrix_data[updated_route[i-1]-1, updated_route[i]-1] + distance_matrix_data[updated_route[i]-1, updated_route[i+1]-1] - distance_matrix_data[updated_route[i-1]-1, updated_route[i+1]-1]

                insertions[tuple(map(tuple, updated_routes))] = cost_difference

    if len(insertions) == 1:
        best_insertion = min(insertions, key=insertions.get)
        return best_insertion, 0

    elif len(insertions) > 1:
        best_insertion = min(insertions, key=insertions.get)

        if len(set(insertions.values())) == 1:  # when all options are of equal value:
            regret = 0
        else:
            regret = sorted(list(insertions.values()))[1] - min(insertions.values())
        return best_insertion, regret
    else:
        # no insertions possible for this customer
        return -1, -1


def regret_insertion(current, rnd_state, prob=1.5, **kwargs):
    visited_customers = [customer for route in current.routes for customer in route]
    all_customers = set(range(1, current.nb_customers + 1))
    unvisited_customers = all_customers - set(visited_customers)

    repaired = copy.deepcopy(current)
    while unvisited_customers:
        insertion_options = {}
        for customer in unvisited_customers:
            best_insertion, regret = get_regret_single_insertion(repaired.routes, customer, repaired.truck_capacity,
                                                                 repaired.dist_matrix_data,
                                                                 repaired.dist_depot_data, repaired.demands_data)
            if best_insertion != -1:
                insertion_options[best_insertion] = regret

        if not insertion_options:
            repaired.routes.append([rnd.choice(list(unvisited_customers))])
        else:
            insertion_option = 0
            while rnd.random() < 1 / prob and insertion_option < len(insertion_options) - 1:
                insertion_option += 1
            repaired.routes = list(map(list, sorted(insertion_options, reverse=True)[insertion_option]))

        visited_customers = [customer for route in repaired.routes for customer in route]
        unvisited_customers = all_customers - set(visited_customers)
    return repaired


def random_repair(current, rnd_state, max_attempts=10, **kwargs):
    print("start repair")
    """
    Simplified Random Repair operator.
    Randomly inserts unassigned customers into the current solution.
    If no feasible insertion exists after a few attempts, a new route is created.
    """
    # repaired = copy.deepcopy(current)

    while current.unassigned:
        customer = current.unassigned.pop()
        inserted = False

        # 尝试在现有路径中插入客户
        for _ in range(max_attempts):
            if not current.routes:
                break  # 如果没有现有路径，跳出循环

            # for route_idx in range(len(current.routes)):
                # total = cvrp_helper_functions.calculate_total_load(current.route, current.tasks_info, current.distance_matrix, current.tank_capacity,
                #                                                     current.now_energy, current.fuel_consumption_rate, current.charging_rate, current.velocity)
                # if cvrp_helper_functions.compute_route_load(routes[route_idx], demands_data) + demands_data[
                #     customer - 1] <= truck_capacity:

            # 随机选择一个路径索引
            route_index = rnd_state.randint(0, len(current.routes))
            route = current.routes[route_index]

            # 随机选择一个插入位置（从1到len(route)，包括在末尾插入）
            insert_position = rnd_state.randint(1, len(route))

            # 创建临时路径以检查可行性
            temp_route = route[:insert_position] + [customer] + route[insert_position:]

            if cvrp_helper_functions.is_nc_feasible(temp_route, current.tasks_info, current.distance_matrix,
                                                    current.tank_capacity, current.now_energy,
                                                    current.fuel_consumption_rate, current.charging_rate,
                                                    current.velocity, current.load_capacity):  # 检查可行性
                current.routes[route_index].insert(insert_position, customer)
                inserted = True
                break  # 成功插入后，退出尝试循环

        # 如果在所有尝试中都未能插入，则创建新路径
        if not inserted:
            new_route = ['D0', customer, 'D0']
            current.routes.append(new_route)

    current = cvrp_helper_functions.process_route(current, current.nodes, current.tasks_info, current.distance_matrix,
                                                  current.tank_capacity, current.now_energy,
                                                  current.fuel_consumption_rate, current.charging_rate,
                                                  current.velocity, current.load_capacity)

    return current
