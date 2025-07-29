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


def time_based_repair(current, rnd_state, max_attempts=5, early_stop_threshold=5, **kwargs):
    """
    Optimized Time-based Repair operator.
    Inserts unassigned customers into the current solution by considering
    the change in the finish time of the route as the insertion cost.
    Includes early stopping and limited search space.
    """
    while current.unassigned:
        best_customer = None
        best_route = None
        best_insert_position = None
        min_increase_in_finish_time = float('inf')

        # 尝试在有限的路径和插入位置中找到最佳插入点
        for customer in current.unassigned:
            for _ in range(max_attempts):
                if not current.routes:
                    break  # 如果没有现有路径，跳出循环

                # 随机选择一个路径索引
                route_index = rnd_state.randint(0, len(current.routes))
                route = current.routes[route_index]

                # 在空路径 ([D0, D0]) 中直接在起点后插入
                if len(route) == 2:
                    insert_position = 1
                else:
                    insert_position = rnd_state.randint(1, len(route) - 1)

                temp_route = route[:insert_position] + [customer] + route[insert_position:]

                # 检查能量与时窗可行性
                if not cvrp_helper_functions.is_nc_feasible(
                    temp_route,
                    current.tasks_info, current.distance_matrix,
                    current.tank_capacity, current.now_energy,
                    current.fuel_consumption_rate, current.charging_rate,
                    current.velocity, current.load_capacity
                ):
                    continue

                # 计算插入前后的完成时间增量
                finish_time_before = cvrp_helper_functions.calculate_arrival_times(
                    route,
                    current.tasks_info, current.distance_matrix,
                    current.tank_capacity, current.now_energy,
                    current.fuel_consumption_rate, current.charging_rate,
                    current.velocity
                )[-1]
                finish_time_after = cvrp_helper_functions.calculate_arrival_times(
                    temp_route,
                    current.tasks_info, current.distance_matrix,
                    current.tank_capacity, current.now_energy,
                    current.fuel_consumption_rate, current.charging_rate,
                    current.velocity
                )[-1]
                increase_in_finish_time = finish_time_after - finish_time_before

                if increase_in_finish_time < min_increase_in_finish_time:
                    min_increase_in_finish_time = increase_in_finish_time
                    best_customer = customer
                    best_route = route
                    best_insert_position = insert_position

                    # 提前终止条件
                    if min_increase_in_finish_time <= early_stop_threshold:
                        break

        # 执行最佳插入或新建路径
        if best_customer is not None:
            best_route.insert(best_insert_position, best_customer)
            current.unassigned.remove(best_customer)
        else:
            customer = current.unassigned.pop()
            new_route = ['D0', customer, 'D0']
            current.routes.append(new_route)

    # 后处理所有路径
    current = cvrp_helper_functions.process_route(current, current.nodes, current.tasks_info, current.distance_matrix,
                                                  current.tank_capacity, current.now_energy,
                                                  current.fuel_consumption_rate, current.charging_rate,
                                                  current.velocity, current.load_capacity)

    return current


def regret_2_insertion(current, rnd_state, regret_threshold=float('inf'), **kwargs):
    """
    Regret-2 Insertion operator.
    Inserts the customer whose cost difference between the best and
    second-best insertion positions is maximal (regret-based).
    """
    while current.unassigned:
        max_regret_value = -1
        best_customer = None
        best_route = None
        best_insert_position = None

        # 对每个未分配客户，计算其前两优插入成本
        for customer in current.unassigned:
            best_cost, second_best_cost, best_pos, route_candidate = find_best_two_insert_positions(customer, current)

            if best_cost is not None and second_best_cost is not None:
                regret_value = second_best_cost - best_cost
            else:
                regret_value = float('inf')

            if regret_value > max_regret_value:
                max_regret_value = regret_value
                best_customer = customer
                best_route = route_candidate
                best_insert_position = best_pos

            if max_regret_value >= regret_threshold:
                break  # 达到阈值提前终止

        # 执行插入或新建路径
        if best_customer is not None:
            if best_route is not None:
                best_route.insert(best_insert_position, best_customer)
            else:
                new_route = ['D0', best_customer, 'D0']
                current.routes.append(new_route)
            current.unassigned.remove(best_customer)
        else:
            break  # 无可行插入，退出

    # 后处理所有路径
    current = cvrp_helper_functions.process_route(
        current, current.nodes, current.tasks_info, current.distance_matrix,
        current.tank_capacity, current.now_energy,
        current.fuel_consumption_rate, current.charging_rate,
        current.velocity, current.load_capacity
    )
    return current


def find_best_two_insert_positions(customer, current):
    """
    Helper: 找出插入指定 customer 的两个最优插入位置及其成本。
    返回 (best_cost, second_best_cost, best_position, best_route)。
    """
    insertion_costs = []

    for route in current.routes:
        for i in range(1, len(route)):
            temp_route = route[:i] + [customer] + route[i:]
            if cvrp_helper_functions.is_nc_feasible(
                temp_route,
                current.tasks_info, current.distance_matrix,
                current.tank_capacity, current.now_energy,
                current.fuel_consumption_rate, current.charging_rate,
                current.velocity, current.load_capacity
            ):
                cost = cvrp_helper_functions.calculate_nc_route_cost(
                    temp_route,
                    current.tasks_info, current.distance_matrix,
                    current.tank_capacity, current.now_energy,
                    current.fuel_consumption_rate, current.charging_rate,
                    current.velocity, current.load_capacity
                )
                insertion_costs.append((cost, i, route))

    # 按成本升序排序，提取最优和次优
    insertion_costs.sort(key=lambda x: x[0])
    if len(insertion_costs) < 2:
        return None, None, None, None

    best_cost, best_position, best_route = insertion_costs[0]
    second_best_cost, _, _ = insertion_costs[1]
    return best_cost, second_best_cost, best_position, best_route


def regret_3_insertion(current, rnd_state, regret_threshold=float('inf'), **kwargs):
    """
    Implements the Regret-3 Insertion heuristic.
    Inserts unassigned customers into the current solution based on the
    difference in insertion cost between the best, second-best, and third-best positions.
    Early stops if a sufficiently high regret value is found.
    """
    while current.unassigned:
        max_regret_value = -1
        best_customer = None
        best_route = None
        best_insert_position = None

        for customer in current.unassigned:
            best_cost, second_best_cost, third_best_cost, best_position, route_candidate = find_best_three_insert_positions(
                customer, current)

            if best_cost is not None and second_best_cost is not None and third_best_cost is not None:
                regret_value = third_best_cost - best_cost
            else:
                regret_value = float('inf')

            if regret_value > max_regret_value:
                max_regret_value = regret_value
                best_customer = customer
                best_route = route_candidate
                best_insert_position = best_position

            if max_regret_value >= regret_threshold:
                break  # 提前终止

        if best_customer is not None:
            if best_route is not None:
                best_route.insert(best_insert_position, best_customer)
            else:
                new_route = ['D0', best_customer, 'D0']
                current.routes.append(new_route)
            current.unassigned.remove(best_customer)
        else:
            break  # 无可行插入，退出

    current = cvrp_helper_functions.process_route(
        current, current.nodes, current.tasks_info, current.distance_matrix,
        current.tank_capacity, current.now_energy,
        current.fuel_consumption_rate, current.charging_rate,
        current.velocity, current.load_capacity
    )
    return current


# def find_best_three_insert_positions(customer, current):
#     """
#     Finds the best, second-best, and third-best insertion positions for the customer.
#     Returns (best_cost, second_best_cost, third_best_cost, best_position, best_route).
#     """
#     insertion_costs = []
#     for route in current.routes:
#         for i in range(1, len(route)):
#             temp_route = route[:i] + [customer] + route[i:]
#             if cvrp_helper_functions.is_nc_feasible(
#                 temp_route,
#                 current.tasks_info, current.distance_matrix,
#                 current.tank_capacity, current.now_energy,
#                 current.fuel_consumption_rate, current.charging_rate,
#                 current.velocity, current.load_capacity
#             ):
#                 cost = cvrp_helper_functions.calculate_nc_route_cost(
#                     temp_route,
#                     current.tasks_info, current.distance_matrix,
#                     current.tank_capacity, current.now_energy,
#                     current.fuel_consumption_rate, current.charging_rate,
#                     current.velocity, current.load_capacity
#                 )
#                 insertion_costs.append((cost, i, route))
#
#     insertion_costs.sort(key=lambda x: x[0])
#     if not insertion_costs:
#         return None, None, None, None, None
#
#     # 提取前三成本，不足时填充为无穷大
#     best_cost, best_position, best_route = insertion_costs[0]
#     second_best_cost = insertion_costs[1][0] if len(insertion_costs) > 1 else float('inf')
#     third_best_cost  = insertion_costs[2][0] if len(insertion_costs) > 2 else float('inf')
#
#     return best_cost, second_best_cost, third_best_cost, best_position, best_route


def find_best_three_insert_positions(customer, current):
    """
    原版动态更新前三成本的方法。
    返回 (best_cost, second_best_cost, third_best_cost, best_position, best_route)。
    """
    best_cost = None
    second_best_cost = None
    third_best_cost = None
    best_position = None
    best_route = None

    for route in current.routes:
        for i in range(1, len(route)):
            temp_route = route[:i] + [customer] + route[i:]
            if not cvrp_helper_functions.is_nc_feasible(
                temp_route,
                current.tasks_info, current.distance_matrix,
                current.tank_capacity, current.now_energy,
                current.fuel_consumption_rate, current.charging_rate,
                current.velocity, current.load_capacity
            ):
                continue

            cost = cvrp_helper_functions.calculate_nc_route_cost(
                temp_route,
                current.tasks_info, current.distance_matrix,
                current.tank_capacity, current.now_energy,
                current.fuel_consumption_rate, current.charging_rate,
                current.velocity, current.load_capacity
            )

            if best_cost is None or cost < best_cost:
                third_best_cost = second_best_cost
                second_best_cost = best_cost
                best_cost = cost
                best_position = i
                best_route = route
            elif second_best_cost is None or cost < second_best_cost:
                third_best_cost = second_best_cost
                second_best_cost = cost
            elif third_best_cost is None or cost < third_best_cost:
                third_best_cost = cost

    if best_cost is None:
        return None, None, None, None, None
    if second_best_cost is None:
        second_best_cost = float('inf')
    if third_best_cost is None:
        third_best_cost = float('inf')

    return best_cost, second_best_cost, third_best_cost, best_position, best_route


def greedy_repair(current, rnd_state, **kwargs):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
    rnd_state.shuffle(current.unassigned)

    while current.unassigned:
        customer = current.unassigned.pop()
        route, idx = best_insert(customer, current)

        if route is not None:
            route.insert(idx, customer)
        else:
            current.routes.append(['D0', customer, 'D0'])

    current = cvrp_helper_functions.process_route(
        current, current.nodes, current.tasks_info, current.distance_matrix,
        current.tank_capacity, current.now_energy,
        current.fuel_consumption_rate, current.charging_rate,
        current.velocity, current.load_capacity
    )
    return current


def best_insert(customer, current):
    """
    Finds the best feasible route and insertion idx for the customer.
    Returns (None, None) if no feasible route insertions are found.
    """
    best_cost = None
    best_route = None
    best_idx = None

    for route in current.routes:
        for idx in range(1, len(route)):
            # 构造临时路径检查可行性
            temp_route = route[:idx] + [customer] + route[idx:]
            if not cvrp_helper_functions.is_nc_feasible(
                temp_route,
                current.tasks_info, current.distance_matrix,
                current.tank_capacity, current.now_energy,
                current.fuel_consumption_rate, current.charging_rate,
                current.velocity, current.load_capacity
            ):
                continue

            cost = insert_cost(customer, temp_route, idx, current.distance_matrix)
            if best_cost is None or cost < best_cost:
                best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx


def insert_cost(customer, route, idx, distance_matrix):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    pre = route[idx - 1] if idx != 0 else 'D0'
    suc = route[idx] if idx != len(route) else 'D0'
    # Increase in cost of adding customer, minus cost of removing old edge
    return distance_matrix[pre][customer] + distance_matrix[customer][suc] - distance_matrix[pre][suc]
