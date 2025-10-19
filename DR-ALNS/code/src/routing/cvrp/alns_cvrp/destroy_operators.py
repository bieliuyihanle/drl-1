import copy
import random
from routing.cvrp.alns_cvrp.cvrp_helper_functions import determine_nr_nodes_to_remove, NormalizeData
from routing.cvrp.alns_cvrp import cvrp_helper_functions

#TODO: put nr_nodes_to_remove in kwargs statement

# --- random removal ---
def random_removal(current, random_state, nr_nodes_to_remove=None):
    # print("random_removal")
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # visited_customers = [customer for route in destroyed_solution.routes for customer in route]
    visited_customers = [c[0] for c in destroyed_solution.customers]

    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    nodes_to_remove = random.sample(visited_customers, nr_nodes_to_remove)
    for node in nodes_to_remove:
        for route in destroyed_solution.routes:
            while node in route:
                route.remove(node)
                visited_customers.remove(node)
                destroyed_solution.unassigned.append(node)

    destroyed_solution.routes = [route for route in destroyed_solution.routes if route != []]

    return destroyed_solution

def worst_dist_cust_removal(current, random_state, nr_nodes_to_remove=None):
    # print("worst_dist_cust_removal")
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # visited_customers = [customer for route in destroyed_solution.routes for customer in route]
    visited_customers = [c[0] for c in destroyed_solution.customers]

    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    cust_cost = []

    for route in destroyed_solution.routes:
        for idx in range(1, len(route) - 1):
            cust_cost_value = cal_cust_cost(route[idx], route, idx, destroyed_solution.distance_matrix)
            cust_cost.append((cust_cost_value, route[idx], route))
    sorted_cust = sorted(cust_cost, key=lambda x: x[0], reverse=True)
    worst_dist_cust = sorted_cust[:nr_nodes_to_remove]

    for _, customer, route in worst_dist_cust:
        route.remove(customer)
        visited_customers.remove(customer)
        destroyed_solution.unassigned.append(customer)

    destroyed_solution.routes = [route for route in destroyed_solution.routes if route != []]

    return destroyed_solution


def cal_cust_cost(customer, route, idx, distance_matrix):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    pre = route[idx - 1] if idx != 0 else 'D0'
    suc = route[idx] if idx != len(route) else 'D0'
    # Increase in cost of adding customer, minus cost of removing old edge
    return distance_matrix[pre][customer] + distance_matrix[customer][suc]


def worst_time_cust_removal(current, random_state, nr_nodes_to_remove=None):
    # print("worst_time_cust_removal")
    """
    Removes customers based on the urgency of their time window.
    It prioritizes removing customers with the most slack time (due_date - arrival_time).
    """
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    visited_customers = [c[0] for c in destroyed_solution.customers]

    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    time_difference = []
    for route_idx, route in enumerate(destroyed_solution.routes):
        vehicle_energy = destroyed_solution.get_vehicle_energy(route_idx)
        arrival_times = cvrp_helper_functions.calculate_arrival_times(
                    route,
                    current.tasks_info, current.distance_matrix,
                    current.tank_capacity, vehicle_energy,
                    current.fuel_consumption_rate, current.charging_rate,
                    current.velocity
                )

        for i in range(1, len(route)):
            if current.tasks_info[route[i]]['Type'] == 'c':
                time_difference_value = current.tasks_info[route[i]]['Due-Date'] - arrival_times[i - 1]
                time_difference.append((time_difference_value, route[i], route))

    sorted_cust = sorted(time_difference, key=lambda x: x[0], reverse=True)
    worst_time_cust = sorted_cust[:nr_nodes_to_remove]

    for _, customer, route in worst_time_cust:
        route.remove(customer)
        visited_customers.remove(customer)
        destroyed_solution.unassigned.append(customer)

    # Rule 4: Use modern, Pythonic syntax
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route]

    return destroyed_solution


def shaw_destroy(current, random_state, nr_nodes_to_remove=None):
    # print("shaw_removal")
    """
    Removes a set of "related" or similar customers from the solution.
    This is also known as the Shaw Removal operator.
    """
    # 1. Safety copy and preparation
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # 2. Externalize control for removal count
    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    # Prepare a clean list of customers currently in the solution
    customers_in_solution = [c[0] for c in destroyed_solution.customers]

    if not customers_in_solution:
        return destroyed_solution

    # 3. Two-Stage Execution: First, "Mark" all nodes to be removed
    to_remove = []

    # Pick the first customer randomly
    first_node = random_state.choice(customers_in_solution)
    to_remove.append(first_node)

    while len(to_remove) < nr_nodes_to_remove:
        # Pick a random node from the set of already removed ones
        last_removed_node = random_state.choice(to_remove)

        # Define candidates for removal
        candidates = [c for c in customers_in_solution if c not in to_remove]
        if not candidates:
            break

        # Calculate relatedness for all candidates to the last removed node
        relatedness_scores = []
        for candidate in candidates:
            # Explicit dependency injection for the relatedness calculation
            score = calculate_relatedness(current, last_removed_node, candidate)
            relatedness_scores.append((score, candidate))

        # Find the most related customer (lowest score)
        if not relatedness_scores:
            break

        most_related_node = min(relatedness_scores, key=lambda x: x[0])[1]
        to_remove.append(most_related_node)

    # 4. Second Stage: "Sweep" (perform the actual removal)
    for node in to_remove:
        for route in destroyed_solution.routes:
            if node in route:
                route.remove(node)
                destroyed_solution.unassigned.append(node)
                break  # Assume customer is in only one route

    # 5. Modern cleanup
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route]

    return destroyed_solution

def calculate_relatedness(current, customer1, customer2):
    # 计算两者之间的距离
    distance_score = current.distance_matrix[customer1][customer2]
    # 根据需求、时间窗等增加其他相关性计算
    route1 = find_route(current, customer1)
    route2 = find_route(current, customer2)
    demand1 = calculate_customer_demand(route1, customer1,current)
    demand2 = calculate_customer_demand(route2, customer2, current)
    demand_score = abs(demand1 - demand2)

    time_window_score = abs(current.tasks_info[customer1]['Due-Date'] - current.tasks_info[customer2]['Due-Date'])

    route_similarity_score = calculate_lij(customer1, customer2, current.routes)

    # 加权组合得到最终的相关性分数
    relatedness_score = (1.0 * distance_score) + (1.0 * demand_score) + (0.5 * time_window_score) + (
                1.0 * route_similarity_score)

    return relatedness_score

def calculate_lij(customer1, customer2, state):
    # 假设self.problem_instance.routes是当前解的路径列表，每个路径包含多个顾客点
    for route in state:
        if customer1 in route and customer2 in route:
            return -1  # 在同一路线中
    return 1  # 不在同一路线中


def calculate_customer_demand(route, customer, current):
    route_idx = current.routes.index(route)
    vehicle_energy = current.get_vehicle_energy(route_idx)
    arrival_times = cvrp_helper_functions.calculate_arrival_times(route, current.tasks_info, current.distance_matrix, current.tank_capacity, vehicle_energy,
                            current.fuel_consumption_rate, current.charging_rate, current.velocity)
    customer_index = route.index(customer)

    customer_demand = (48 - current.tasks_info[route[customer_index]]['stock-at-call-time'] +
                      (arrival_times[customer_index] - current.tasks_info[route[customer_index]]['call-time']) / 30) * 0.75

    return customer_demand


def proximity_based_removal(current, random_state, nr_nodes_to_remove=None):
    # print("proximity_based_removal")
    # 1. Safety copy and preparation
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # 2. Externalize control for removal count
    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    # Prepare a clean list of customers currently in the solution
    customers_in_solution = [c[0] for c in destroyed_solution.customers]

    if not customers_in_solution:
        return destroyed_solution

    # 3. Two-Stage Execution: First, "Mark" all nodes to be removed
    to_remove = []

    # Randomly select the first customer
    first_node = random_state.choice(customers_in_solution)
    to_remove.append(first_node)
    last_selected_node = first_node

    while len(to_remove) < nr_nodes_to_remove:
        # Define candidates for removal (those not already marked)
        candidates = [c for c in customers_in_solution if c not in to_remove]
        if not candidates:
            break

        # Find the customer in the candidate list nearest to the last-removed one.
        # This uses the data-centric distance matrix directly.
        nearest_node = min(
            candidates,
            key=lambda candidate: current.distance_matrix[last_selected_node][candidate]
        )

        to_remove.append(nearest_node)
        last_selected_node = nearest_node

    # 4. Second Stage: "Sweep" (perform the actual removal)
    for node in to_remove:
        for route in destroyed_solution.routes:
            if node in route:
                route.remove(node)
                destroyed_solution.unassigned.append(node)
                break  # Assume customer is in only one route

    # 5. Modern cleanup
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route]

    return destroyed_solution


def time_based_removal(current, random_state, nr_nodes_to_remove=None):
    # print("time_based_removal")
    """
    Removes a chain of customers that are close in time.
    It starts with a random customer, then removes the one whose call time is
    closest to it, then the one closest to that one, and so on.
    """
    # 1. Safety copy and preparation
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # 2. Externalize control for removal count
    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    # Prepare a clean list of customers currently in the solution
    customers_in_solution = [c[0] for c in destroyed_solution.customers]

    if not customers_in_solution:
        return destroyed_solution

    # 3. Two-Stage Execution: First, "Mark" all nodes to be removed
    to_remove = []

    # Randomly select the first customer
    first_node = random_state.choice(customers_in_solution)
    to_remove.append(first_node)
    last_selected_node = first_node

    while len(to_remove) < nr_nodes_to_remove:
        # Define candidates for removal (those not already marked)
        candidates = [c for c in customers_in_solution if c not in to_remove]
        if not candidates:
            break

        # Find the customer in the candidate list with the closest call time
        # This uses the data-centric 'tasks_info' dictionary directly.
        nearest_time_node = min(
            candidates,
            key=lambda candidate: abs(current.tasks_info[last_selected_node]['call-time'] -
                                      current.tasks_info[candidate]['call-time'])
        )

        to_remove.append(nearest_time_node)
        last_selected_node = nearest_time_node

    # 4. Second Stage: "Sweep" (perform the actual removal)
    for node in to_remove:
        for route in destroyed_solution.routes:
            if node in route:
                route.remove(node)
                destroyed_solution.unassigned.append(node)
                break  # Assume customer is in only one route

    # 5. Modern cleanup
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route]

    return destroyed_solution


def zone_removal(current_solution, random_state, nr_nodes_to_remove=None, zone_size=30):
    # print("zone_removal")
    """
    Removes customers that fall within a randomly generated geographic zone.
    The process is repeated with new zones until enough customers are removed.
    """
    # 1. Safety copy and preparation
    destroyed_solution = copy.deepcopy(current_solution)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # 2. Externalize control for removal count
    nb_customers = len(destroyed_solution.customers)
    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(nb_customers)

    # 3. Prepare data once
    customers_in_solution = [c[0] for c in destroyed_solution.customers]
    if not customers_in_solution:
        return destroyed_solution

    # Calculate overall coordinate bounds once for efficiency
    coords = [(current_solution.tasks_info[c]['x'], current_solution.tasks_info[c]['y']) for c in customers_in_solution]
    min_x, min_y = min(c[0] for c in coords), min(c[1] for c in coords)
    max_x, max_y = max(c[0] for c in coords), max(c[1] for c in coords)

    # 4. Two-Stage Execution: First, "Mark" all nodes to be removed
    to_remove = set()

    # Add a break condition to prevent potential infinite loops
    max_tries = 10
    tries = 0
    while len(to_remove) < nr_nodes_to_remove and tries < max_tries:
        # Define a new random zone
        x_lower = random_state.uniform(min_x, max_x - zone_size)
        y_lower = random_state.uniform(min_y, max_y - zone_size)
        x_upper = x_lower + zone_size
        y_upper = y_lower + zone_size

        # Identify customers in the zone who are not already marked for removal
        candidates = [c for c in customers_in_solution if c not in to_remove]
        customers_in_zone = [
            c for c in candidates
            if x_lower <= current_solution.tasks_info[c]['x'] <= x_upper and
               y_lower <= current_solution.tasks_info[c]['y'] <= y_upper
        ]

        if not customers_in_zone:
            tries += 1
            continue

        # Add found customers to the removal set
        to_remove.update(customers_in_zone)
        tries = 0  # Reset tries if we successfully find customers

    # 5. Second Stage: "Sweep" (perform the actual removal)
    # Ensure we don't remove more than intended if the last zone was large
    final_to_remove = list(to_remove)[:nr_nodes_to_remove]

    for node in final_to_remove:
        for route in destroyed_solution.routes:
            if node in route:
                route.remove(node)
                destroyed_solution.unassigned.append(node)
                break

    # 6. Modern cleanup
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route]

    return destroyed_solution


def shortest_route_removal(current, random_state, nr_nodes_to_remove=None):
    # print("shortest_route_removal")
    """
    Finds the route with the shortest total distance and removes all of its
    customers. This is a "chunk" removal operator designed to escape local optima.
    """
    # 1. Safety copy and preparation
    destroyed_solution = copy.deepcopy(current)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    # Filter out empty routes that might exist before processing
    active_routes = [r for r in destroyed_solution.routes if r]

    # 2. Use modern Pythonic syntax to find the shortest route in one line
    # This is more direct than manual iteration.
    shortest_route = min(
        active_routes,
        key=lambda r: cvrp_helper_functions.calculate_route_distance(r, current.distance_matrix)
    )

    # 3. Add all customers from the shortest route to the unassigned list
    # The [1:-1] slice assumes the route starts and ends with a depot (node 0)
    customers_to_remove = shortest_route[1:-1]
    destroyed_solution.unassigned.extend(customers_to_remove)

    # 4. Rebuild the routes list, implicitly removing the shortest route
    destroyed_solution.routes = [r for r in destroyed_solution.routes if r is not shortest_route]

    return destroyed_solution


def least_cus_route_removal(current_solution, random_state, nr_nodes_to_remove=None):
    # print("least_cus_route_removal")
    destroyed_solution = copy.deepcopy(current_solution)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    active_routes = [r for r in destroyed_solution.routes if r]

    least_cus_route = min(active_routes, key=len)

    customers_to_remove = least_cus_route[1:-1]
    destroyed_solution.unassigned.extend(customers_to_remove)

    destroyed_solution.routes = [r for r in destroyed_solution.routes if r is not least_cus_route]

    return destroyed_solution


def random_route_removal(current_solution, random_state, nr_nodes_to_remove=None):
    # print("random_route_removal")
    destroyed_solution = copy.deepcopy(current_solution)

    for route in destroyed_solution.routes:
        route = remove_charging_station(route, destroyed_solution.tasks_info)

    active_routes = [r for r in destroyed_solution.routes if r]

    if not active_routes:
        return destroyed_solution

    route_idx = random_state.randint(len(active_routes))
    route_to_remove = active_routes[route_idx]

    customers_to_remove = route_to_remove[1:-1]
    destroyed_solution.unassigned.extend(customers_to_remove)

    destroyed_solution.routes = [r for r in destroyed_solution.routes if r is not route_to_remove]

    return destroyed_solution

# --- relatedness destroy method ---

# see: Shaw - Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems
# see: Santini, Ropke - A comparison of acceptance criteria for the adaptive large neighbourhood search metaheuristic


def relatedness_removal(current, random_state, nr_nodes_to_remove=None, prob=5):
    # print("relatedness_removal")
    destroyed_solution = copy.deepcopy(current)
    visited_customers = [customer for route in destroyed_solution.routes for customer in route]

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    node_to_remove = random_state.choice(visited_customers)
    for route in destroyed_solution.routes:
        while node_to_remove in route:
            route.remove(node_to_remove)
            visited_customers.remove(node_to_remove)

    for i in range(nr_nodes_to_remove - 1):
        related_nodes = []
        normalized_distances = NormalizeData(destroyed_solution.dist_matrix_data[node_to_remove - 1])
        route_node_to_remove = [route for route in current.routes if node_to_remove in route][0]
        for route in destroyed_solution.routes:
            for node in route:
                if node in route_node_to_remove:
                    related_nodes.append((node, normalized_distances[node - 1]))
                else:
                    related_nodes.append((node, normalized_distances[node - 1] + 1))

        if random_state.random() < 1 / prob:
            node_to_remove = random_state.choice(visited_customers)
        else:
            node_to_remove = min(related_nodes, key=lambda x: x[1])[0]
        for route in destroyed_solution.routes:
            while node_to_remove in route:
                route.remove(node_to_remove)
                visited_customers.remove(node_to_remove)
    destroyed_solution.routes = [route for route in destroyed_solution.routes if route != []]

    return destroyed_solution



# --- neighbor/history graph removal
# see: A unified heuristic for a large class of Vehicle Routing Problems with Backhauls
def neighbor_graph_removal(current, random_state, nr_nodes_to_remove=None, prob=5):
    destroyed_solution = copy.deepcopy(current)

    if nr_nodes_to_remove is None:
        nr_nodes_to_remove = determine_nr_nodes_to_remove(destroyed_solution.nb_customers)

    values = {}
    for route in destroyed_solution.routes:
        if len(route) == 1:
            values[route[0]] = current.graph.get_edge_weight(0, route[0]) + current.graph.get_edge_weight(route[0], 0)
        else:
            for i in range(len(route)):
                if i == 0:
                    values[route[i]] = current.graph.get_edge_weight(0, route[i]) + current.graph.get_edge_weight(
                        route[i], route[1])
                elif i == len(route) - 1:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                      route[i]) + current.graph.get_edge_weight(
                        route[i], 0)
                else:
                    values[route[i]] = current.graph.get_edge_weight(route[i - 1],
                                                                      route[i]) + current.graph.get_edge_weight(
                        route[i], route[i + 1])

    removed_nodes = []
    while len(removed_nodes) < nr_nodes_to_remove:
        # sort the nodes based on their neighbor graph scores in descending order
        sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)
        # select the node to remove
        removal_option = 0
        while random_state.random() < 1 / prob and removal_option < len(sorted_nodes) - 1:
            removal_option += 1
        node_to_remove, score = sorted_nodes[removal_option]

        # remove the node from its route
        for route in destroyed_solution.routes:
            if node_to_remove in route:
                route.remove(node_to_remove)
                removed_nodes.append(node_to_remove)

                values.pop(node_to_remove)
                if len(route) == 0:
                    destroyed_solution.routes.remove([])

                elif len(route) == 1:
                    values[route[0]] = current.graph.get_edge_weight(0, route[0]) + current.graph.get_edge_weight(
                        route[0], 0)
                else:
                    for i in range(len(route)):
                        if i == 0:
                            values[route[i]] = current.graph.get_edge_weight(0, route[
                                i]) + current.graph.get_edge_weight(route[i], route[1])
                        elif i == len(route) - 1:
                            values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                                i]) + current.graph.get_edge_weight(route[i], 0)
                        else:
                            values[route[i]] = current.graph.get_edge_weight(route[i - 1], route[
                                i]) + current.graph.get_edge_weight(route[i], route[i + 1])

                break

    return destroyed_solution

def find_route(current, customer):
    """
    Return the route that contains the passed-in customer.
    """
    for route in current.routes:
        if customer in route:
            return route

    raise ValueError(f"Solution does not contain customer {customer}.")

def remove_charging_station(route, tasks_info):
    route[:] = [node for node in route if tasks_info[node]['Type'] != 'f']
    return route