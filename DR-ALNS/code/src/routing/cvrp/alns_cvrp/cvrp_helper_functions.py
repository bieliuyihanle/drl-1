import pandas as pd
import sys
import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple


def read_elem(filename):
    with open(filename) as f:
        return [str(elem) for elem in f.read().split()]


# def read_input_cvrp(filename, instance_nr):
#     data = pd.read_pickle(filename)
#     depot_x = data[instance_nr][0][0]
#     depot_y = data[instance_nr][0][1]
#     customers_x = [x for x,y in data[instance_nr][1]]
#     customers_y = [y for x,y in data[instance_nr][1]]
#     demands = data[instance_nr][2]
#     capacity = data[instance_nr][3]
#
#     distance_matrix = compute_distance_matrix(customers_x, customers_y)
#     distance_depots = compute_distance_depots(depot_x, depot_y, customers_x, customers_y)
#
#     return len(demands), capacity, distance_matrix, distance_depots, demands


def load_problem_instance(file):
    with (open(file) as f):
        f.readline()  # ignore header

        target_line = f.readline()

        customers = []
        fuel_stations = []
        depot = None

        while target_line != '\n':
            stl = target_line.split()  # splitted target_line
            idx = int(stl[0][1:])

            new_target = [stl[0], idx, float(stl[2]), float(stl[3]), int(float(stl[4])),
                          int(float(stl[5])), int(float(stl[6])), int(float(stl[7]))]

            if stl[1] == 'd':
                depot = new_target
            elif stl[1] == 'f':
                fuel_stations.append(new_target)
            elif stl[1] == 'c':
                customers.append(new_target)

            target_line = f.readline()

        configuration_line = f.readline()
        tank_capacity = float(configuration_line.split('/')[1])  # q Vehicle fuel tank capacity

        configuration_line = f.readline()
        now_energy = float(configuration_line.split('/')[1])  # e Vehicle now fuel tank capacity

        configuration_line = f.readline()
        load_capacity = float(configuration_line.split('/')[1])  # C Vehicle load capacity

        configuration_line = f.readline()
        fuel_consumption_rate = float(configuration_line.split('/')[1])  # r fuel consumption rate

        configuration_line = f.readline()
        charging_rate = float(configuration_line.split('/')[1])  # g inverse refueling rate

        configuration_line = f.readline()
        velocity = float(configuration_line.split('/')[1])  # v average Velocity

        nodes, tasks_info = build_tasks_info(depot, customers, fuel_stations)
        distance_matrix = compute_dist_matrix(nodes, tasks_info)

        return tank_capacity, now_energy, load_capacity, fuel_consumption_rate, charging_rate, velocity, depot, \
               customers, fuel_stations, nodes, tasks_info, distance_matrix


def build_tasks_info(
        depot: List[Any],
        customers: List[List[Any]],
        fuel_stations: List[List[Any]]
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    构建任务信息字典和节点名称列表

    返回：
      - nodes: 按顺序的节点名称列表
      - tasks_info: { name -> 属性字典 }
    """
    tasks_info: Dict[str, Dict[str, Any]] = {}

    # depot
    tasks_info[depot[0]] = {
        'Type': 'd',
        'x': depot[2],
        'y': depot[3],
        'stock-at-call-time': depot[4],
        'call-time': depot[5],
        'Due-Date': depot[6],
        'Service-Time': depot[7],
    }

    # customers
    for c in customers:
        tasks_info[c[0]] = {
            'Type': 'c',
            'x': c[2],
            'y': c[3],
            'stock-at-call-time': c[4],
            'call-time': c[5],
            'Due-Date': c[6],
            'Service-Time': c[7],
        }

    # fuel stations
    for s in fuel_stations:
        tasks_info[s[0]] = {
            'Type': 'f',
            'x': s[2],
            'y': s[3],
            'stock-at-call-time': s[4],
            'call-time': s[5],
            'Due-Date': s[6],
            'Service-Time': s[7],
        }

    # 保证顺序：depot, customers..., stations...
    nodes = [depot[0]] + [c[0] for c in customers] + [s[0] for s in fuel_stations]
    return nodes, tasks_info


def compute_dist_matrix(
        nodes: List[str],
        tasks_info: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    根据 nodes 顺序和 tasks_info 中的 (x,y)，计算 dict-of-dict 格式的距离矩阵
    """
    N = len(nodes)
    # 先提取按顺序的坐标元组列表
    coords = [(tasks_info[name]['x'], tasks_info[name]['y']) for name in nodes]

    # 初始化 raw_dist
    raw = [[0.0] * N for _ in range(N)]
    for i in range(N):
        xi, yi = coords[i]
        for j in range(i + 1, N):
            xj, yj = coords[j]
            d = abs(xi - xj) + abs(yi - yj)
            raw[i][j] = raw[j][i] = d

    # 转成更易用的 dict-of-dict
    dist_matrix: Dict[str, Dict[str, float]] = {
        u: {v: raw[i][j] for j, v in enumerate(nodes)}
        for i, u in enumerate(nodes)
    }
    return dist_matrix


# Compute the distance matrix
# def compute_distance_matrix(depot, customers, fuel_stations):
#     customers_id = [a[0] for a in customers]
#     fuel_stations_id = [b[0] for b in fuel_stations]
#     nodes = [depot[0]] + customers_id + fuel_stations_id
#
#     nb_customers = len(customers)
#     distance_matrix = np.zeros((nb_customers, nb_customers))
#     for i in range(nb_customers):
#         distance_matrix[i][i] = 0
#         for j in range(nb_customers):
#             dist = compute_dist(customers[i][2], customers[j][2], customers[i][3], customers[j][3])
#             distance_matrix[i][j] = dist
#             distance_matrix[j][i] = dist
#     return distance_matrix


def is_nc_feasible(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    if tw_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        return False
    elif payload_capacity_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        return False
    else:
        return True


def is_feasible(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    if tw_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        return False
    elif tank_capacity_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        return False
    elif payload_capacity_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        return False
    else:
        return True


def is_complete(route, tasks_info):
    return tasks_info[route[0]]['Type'] == 'd' and tasks_info[route[-1]]['Type'] == 'd' \
            and all(tasks_info[node]['Type'] != 'd' for node in route[1:-1])


def tw_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    elapsed_time = tasks_info[route[0]]['call-time']

    for i in range(1, len(route)):
        elapsed_time += distance_matrix[route[i - 1]][route[i]] / velocity
        if elapsed_time > tasks_info[route[i]]['Due-Date']:
            return True

        if tasks_info[route[i]]['Type'] is 'f':
            missing_energy = tank_capacity - calculate_remaining_tank(route, tasks_info, distance_matrix,
                                                                      tank_capacity, now_energy,
                                                                      fuel_consumption_rate, until=i)
            tasks_info[i]['Service-Time'] = missing_energy * charging_rate

        elapsed_time += tasks_info[i]['Service-Time']

    return False


def payload_capacity_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                                         fuel_consumption_rate, charging_rate, velocity, load_capacity):

    total_demand = calculate_total_load(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                                         fuel_consumption_rate, charging_rate, velocity)

    return total_demand > load_capacity


def tank_capacity_constraint_violated(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    last = None

    for t in route:
        if last is not None:
            consumption = distance_matrix[last][t] * fuel_consumption_rate
            now_energy -= consumption

            if tank_capacity < 0:
                return True

            if tasks_info[route[t]]['Type'] is 'f':
                now_energy = tank_capacity
        last = t

    return False


def calculate_arrival_times(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity):
    elapsed_time = tasks_info['D0']['call-time']
    last = None
    arrival_times = [elapsed_time]

    for i in route:
        if last is not None:
            travel_time = distance_matrix[last][i] / velocity
            elapsed_time += travel_time
            i.arrival_time = elapsed_time    # 此处后续报错可直接删除

            # 记录当前节点的实际到达时间
            arrival_times.append(elapsed_time)

            if tasks_info[i]['Type'] is 'f':
                missing_energy = tank_capacity - calculate_remaining_tank(route, tasks_info, distance_matrix,
                                                                          tank_capacity, now_energy,
                                                                          fuel_consumption_rate, until=i)
                tasks_info[i]['Service-Time'] = missing_energy * charging_rate

            elapsed_time += tasks_info[i]['Service-Time']

        last = i
    return arrival_times


def calculate_remaining_tank(route, tasks_info, distance_matrix, tank_capacity, now_energy,fuel_consumption_rate,
                             until=None):
    last = None
    now_energy = now_energy
    total_consumption = 0
    for t in route:
        if last is not None:
            distance = distance_matrix[last][t]
            consumption = distance * fuel_consumption_rate
            # total_consumption += consumption
            now_energy -= consumption

            if tasks_info[t]['Type'] is 'f':
                now_energy = tank_capacity

            if until == t:
                return now_energy

        last = t
        # print(total_consumption)
    return now_energy


def calculate_demand(route, tasks_info, distance_matrix, tank_capacity, now_energy, fuel_consumption_rate,
                     charging_rate, velocity):
    arrival_times = calculate_arrival_times(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity)
    demand = []

    for i in range(len(route)):
        if tasks_info[route[i]]['Type'] is 'f':
            tasks_demand = 0
            demand.append(tasks_demand)
        elif tasks_info[route[i]]['Type'] is 'c':
            tasks_demand = (48 - tasks_info[route[i]]['stock-at-call-time'] + (arrival_times[i] -
                                                                             tasks_info[route[i]]['call-time']) / 30) * 0.75
            demand.append(tasks_demand)
        elif tasks_info[route[i]]['Type'] is 'd':
            tasks_demand = 0
            demand.append(tasks_demand)
    return demand


def calculate_total_load(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                                         fuel_consumption_rate, charging_rate, velocity):
    demand = calculate_demand(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                              fuel_consumption_rate, charging_rate, velocity)

    total_demand = sum(demand)
    return total_demand


def remove_charging_station(route, tasks_info):
    route[:] = [node for node in route if tasks_info[node]['Type'] != 'f']
    return route


def need_charge(route, distance_matrix, now_energy, fuel_consumption_rate, tank_capacity):
    return now_energy - calculate_route_distance(route, distance_matrix) * fuel_consumption_rate < 0.2 * tank_capacity


def calculate_route_distance(route, distance_matrix):
    dist = 0
    last = None
    for r in route:
        if last is not None:
            dist += distance_matrix[last][r]
        last = r
    return dist


def get_reachable_charging_stations(route, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                    fuel_consumption_rate, until = None) -> list:
    capacity = calculate_remaining_tank(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                                        fuel_consumption_rate, until)
    max_dist = capacity / fuel_consumption_rate
    reachable_stations = []

    charging_stations = [n for n in nodes if tasks_info[n]['Type'] == 'f']
    for cs in charging_stations:
        if distance_matrix[cs][until] <= max_dist:
            reachable_stations.append(cs)

    return reachable_stations


def find_optimal_charging_station_insertion(route, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    # print(route)
    best_insertion_point = None
    best_station = None
    min_route_cost = float('inf')

    for i in range(1, len(route)):
        if tasks_info[route[i]]['Type'] is not 'f':
            reachable_stations = get_reachable_charging_stations(route[:i], nodes, tasks_info, distance_matrix,
                                                                 tank_capacity, now_energy, fuel_consumption_rate,
                                                                 until=i)
            if reachable_stations is None:
                continue

            else:
                for j in reachable_stations:
                    temp_route = route[:i] + [j] + route[i:]

                    # 判断插入后路径是否可行
                    if is_feasible(temp_route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):

                        # 计算插入后的总成本
                        route_cost = calculate_route_cost(temp_route, tasks_info, distance_matrix, tank_capacity,
                                                          now_energy, fuel_consumption_rate, charging_rate, velocity,
                                                          load_capacity)
                        # Tr = self.calculate_arrival_times(temp_route)
                        # print(Tr)
                        # print(temp_route)
                        # print(route_cost)
                        # 更新最优插入点和总成本
                        if route_cost < min_route_cost:
                            min_route_cost = route_cost
                            best_insertion_point = i
                            best_station = j
    # print(best_insertion_point, best_station, min_route_cost)
    return best_insertion_point, best_station, min_route_cost


def make_route_feasible_and_best(route, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    best_insertion_point, best_station, min_route_cost = find_optimal_charging_station_insertion(route, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity)
    if best_station == None:
        return None
    best_feasible_route = route[:best_insertion_point]+[best_station]+route[best_insertion_point:]

    # print(best_feasible_route)
    if not is_feasible(best_feasible_route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        # print("NO")
        return None
    else:
        # print("YES")
        # print(best_feasible_route)
        return best_feasible_route


def process_route(state, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    unvisited_customers = []

    for idx, value in enumerate(state.state):

        # 判断当前路径是否需要充电
        while need_charge(state.state[idx], distance_matrix, now_energy, fuel_consumption_rate, tank_capacity):

            k = make_route_feasible_and_best(value, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity)
            if k is not None:
                # 找到可行的插入点，更新路径
                state.state[idx] = k
                break  # 找到充电位置，跳出while循环，继续下一条路径
            else:
                # 找不到插入点，移除最小到达时间的客户点
                earliest_customer = find_earliest_customer(state.state[idx], tasks_info)
                state.state[idx].remove(earliest_customer)    # 从当前路径中移除
                unvisited_customers.append(earliest_customer)  # 将该客户点标记为未访问
                # 路径调整后，继续判断是否需要充电
    while unvisited_customers:
        # print("Current unvisited_customers:", unvisited_customers)

        # 使用k-最近邻域启发式为未访问的客户生成新的路径
        new_routes = innh(unvisited_customers, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity, k=3)
        unvisited_customers = []
        # print(new_routes)
        # 对新的路径进行充电站插入操作
        for idx, value in enumerate(new_routes):
            # print(value)
            # print(self.need_charge(value))
            while need_charge(value, distance_matrix, now_energy, fuel_consumption_rate, tank_capacity):
                k = make_route_feasible_and_best(value, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity)
                if k is not None:
                    new_routes[idx] = k
                    break
                else:
                    earliest_customer = find_earliest_customer(new_routes[idx], tasks_info)
                    new_routes[idx].remove(earliest_customer)
                    unvisited_customers.append(earliest_customer)

        # 展开 new_routes，将其转化为单个客户的列表
        flattened_new_routes = [customer for route in new_routes for customer in route]
        # print(flattened_new_routes)
        # 更新 unvisited_customers 列表，确保未服务的客户仍然保留

        unvisited_customers = [customer for customer in unvisited_customers if customer not in flattened_new_routes]

        # print("11111111Current unvisited_customers:", unvisited_customers)
        state.state.extend(new_routes)
    # print("22222222222222")
    return state


def innh(customers, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity, k=3):
    giant_route = []
    serviced_customers = set()
    unserved_customers = customers.copy()

    while unserved_customers:
        route = ['D0']
        last_position = 'D0'

        while unserved_customers:
            possible_successors = [customer for customer in unserved_customers if customer not in serviced_customers]
            possible_successors.sort(key=lambda n: distance_matrix[last_position][n])
            possible_successors = possible_successors[:k]
            if not possible_successors:
                break
            successor = min(possible_successors, key=lambda n: tasks_info[n]['Due-Date'])
            route.append(successor)
            # demand = route.calculate_demand()
            if not is_nc_feasible(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
                route.pop()
                break
            serviced_customers.add(successor)
            unserved_customers.remove(successor)
            last_position = successor
        route.append('D0')
        if len(route) > 1:
            giant_route.append(route)
            unserved_customers = [customer for customer in unserved_customers if customer not in serviced_customers]

    return giant_route


def find_earliest_customer(route, tasks_info):
    # 假设每个客户点包含属性 'due_date'
    # 返回路径中 due_date 最小的客户点
    earliest_customer = None
    min_due_date = float('inf')

    for v in route:
        if tasks_info[v]['Type'] is 'c':
            if tasks_info[v]['Due-Date'] < min_due_date:
                min_due_date = tasks_info[v]['Due-Date']
                earliest_customer = v

    return earliest_customer


def calculate_route_cost(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                         fuel_consumption_rate, charging_rate, velocity, load_capacity):
    route_cost = 0
    if not is_feasible(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                         fuel_consumption_rate, charging_rate, velocity, load_capacity):
        route_cost = float('inf')
    else:
        dist_cost = calculate_route_distance(route, distance_matrix)
        # print(dist_cost)
        arrival_times = calculate_arrival_times(route, tasks_info, distance_matrix, tank_capacity, now_energy,
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

    return route_cost


def get_nb_trucks(filename):
    begin = filename.rfind("-k")
    if begin != -1:
        begin += 2
        end = filename.find(".", begin)
        return int(filename[begin:end])
    print("Error: nb_trucks could not be read from the file name. Enter it from the command line")
    sys.exit(1)


def compute_route_load(route, demands_data):
    load = 0
    for i in route:
        load += demands_data[i - 1]
    return load


def get_customers_that_can_be_added_to_route(route_load, truck_capacity, unvisited_customers, demands_data):
    unvisited_edgible_customers = []
    for customer in unvisited_customers:
        if route_load + demands_data[customer - 1] <= truck_capacity:
            unvisited_edgible_customers.append(customer)
    return unvisited_edgible_customers


def get_closest_customer_to_add(route, unvisited_edgible_customers, dist_matrix_data, dist_depot_data):
    current_node = route[-1]
    distances = [dist_matrix_data[current_node - 1][unvisited_node - 1] for unvisited_node in
                 unvisited_edgible_customers]
    closest_customer = unvisited_edgible_customers[
        pd.Series(distances).idxmin()]  # NOTE: no -1 because this is an index, not an id
    return closest_customer


def cost_routes(routes, dist_matrix_data, distance_depot_data):
    cost = 0
    for route in routes:
        cost += distance_depot_data[route[0] - 1] + distance_depot_data[route[-1] - 1]
        for i in range(len(route) - 1):
            cost += dist_matrix_data[route[i] - 1][route[i + 1] - 1]
    return cost


def determine_nr_nodes_to_remove(nb_customers, omega_bar_minus=5, omega_minus=0.1, omega_bar_plus=50, omega_plus=0.4):
    n_plus = min(omega_bar_plus, omega_plus * nb_customers)
    n_minus = min(n_plus, max(omega_bar_minus, omega_minus * nb_customers))
    r = random.randint(round(n_minus), round(n_plus))
    return r


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def update_neighbor_graph(current, new_routes, new_routes_quality):
    for route in new_routes:
        prev_node = 0
        for i in range(len(route)):
            curr_node = route[i]
            prev_edge_weight = current.graph.get_edge_weight(prev_node, curr_node)
            if new_routes_quality < prev_edge_weight:
                current.graph.update_edge(prev_node, curr_node, new_routes_quality)
            prev_node = curr_node
        prev_edge_weight = current.graph.get_edge_weight(prev_node, 0)
        if new_routes_quality < prev_edge_weight:
            current.graph.update_edge(prev_node, 0, new_routes_quality)
    return current.graph


class NeighborGraph:
    def __init__(self, num_nodes):
        self.graph = np.full((num_nodes + 1, num_nodes + 1), np.inf, dtype=np.float64)

    def update_edge(self, node_a, node_b, cost):
        # graph is kept single directional
        self.graph[node_a][node_b] = cost

    def get_edge_weight(self, node_a, node_b):
        return self.graph[node_a][node_b]
