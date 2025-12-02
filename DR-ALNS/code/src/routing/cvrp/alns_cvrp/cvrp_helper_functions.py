import re
import pandas as pd
import sys
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path


def _evaluate_route_metrics(
        route: Sequence[Any],
        route_index: int,
        tasks_info,
        distance_matrix,
        tank_capacity,
        now_energy,
        fuel_consumption_rate,
        charging_rate,
        velocity,
        load_capacity: Optional[float] = None,
):
    """Internal helper that computes detailed metrics for a single route."""

    initial_energy = get_vehicle_energy(now_energy, route_index, tank_capacity)

    if not route:
        remaining_energy = float(initial_energy)
        return {
            'index': route_index,
            'route': [],
            'initial_energy': float(initial_energy),
            'remaining_energy': remaining_energy,
            'distance_cost': 0.0,
            'time_cost': 0.0,
            'route_cost': 0.0,
            'is_feasible': True if load_capacity is not None else None,
        }

    dist_cost = calculate_route_distance(route, distance_matrix)
    arrival_times = calculate_arrival_times(
        route,
        tasks_info,
        distance_matrix,
        tank_capacity,
        initial_energy,
        fuel_consumption_rate,
        charging_rate,
        velocity,
    )

    time_cost = 0.0
    for i in range(1, len(route)):
        node = route[i]
        if tasks_info[node]['Type'] == 'c':
            time_cost += tasks_info[node]['Due-Date'] - arrival_times[i]

    route_cost = dist_cost + 0.5 * time_cost + 200

    remaining_energy = calculate_remaining_tank(
        route,
        tasks_info,
        distance_matrix,
        tank_capacity,
        initial_energy,
        fuel_consumption_rate,
    )

    feasible: Optional[bool] = None
    if load_capacity is not None:
        feasible = is_feasible(
            route,
            tasks_info,
            distance_matrix,
            tank_capacity,
            initial_energy,
            fuel_consumption_rate,
            charging_rate,
            velocity,
            load_capacity,
        )

    return {
        'index': route_index,
        'route': list(route),
        'initial_energy': float(initial_energy),
        'remaining_energy': float(remaining_energy),
        'distance_cost': float(dist_cost),
        'time_cost': float(time_cost),
        'route_cost': float(route_cost),
        'is_feasible': feasible,
    }


def evaluate_routes_with_energy(
        routes: Sequence[Sequence[Any]],
        tasks_info,
        distance_matrix,
        tank_capacity,
        now_energy,
        fuel_consumption_rate,
        charging_rate,
        velocity,
        load_capacity: Optional[float] = None,
):
    """Evaluate manually supplied routes and return cost and energy details.

    Parameters
    ----------
    routes:
        A list of routes, where each route is an ordered list of node identifiers.
    tasks_info, distance_matrix, tank_capacity, fuel_consumption_rate, charging_rate, velocity:
        Problem data describing the instance.
    now_energy:
        Initial energy configuration. Accepts a scalar, sequence or dictionary, in the
        same format as other helpers in this module.
    load_capacity:
        Optional vehicle load capacity. When provided, a feasibility flag is computed
        for each route using the existing constraint checks.

    Returns
    -------
    dict
        A dictionary containing the total cost, the per-route breakdown and the
        remaining energy for each vehicle. This makes it easy to check the cost of
        user-provided solutions period by period.
    """

    route_summaries = []
    total_cost = 0.0

    for idx, route in enumerate(routes):
        summary = _evaluate_route_metrics(
            route,
            idx,
            tasks_info,
            distance_matrix,
            tank_capacity,
            now_energy,
            fuel_consumption_rate,
            charging_rate,
            velocity,
            load_capacity,
        )
        route_summaries.append(summary)
        total_cost += summary['route_cost']

    remaining_energies = [summary['remaining_energy'] for summary in route_summaries]

    return {
        'total_cost': float(total_cost),
        'route_summaries': route_summaries,
        'remaining_energies': remaining_energies,
    }


def evaluate_solution_cost_and_energy(
        routes: Sequence[Sequence[Any]],
        *,
        tasks_info,
        distance_matrix,
        tank_capacity: float,
        initial_energy: Union[float, Sequence[float], Dict[int, float]],
        fuel_consumption_rate: float,
        charging_rate: float,
        velocity: float,
        load_capacity: Optional[float] = None,
):
    """Convenience wrapper that exposes cost/energy computation for custom routes.

    这个函数可以直接用于手动指定的解：

    ``routes``
        每一辆车的路径（如 ``[["d0", "c1", "c2", "d0"], ...]``）。
    ``tasks_info`` 和 ``distance_matrix``
        来自实例文件（例如通过 :func:`build_tasks_info` 构造）。
    其余参数
        为车辆的相关属性：油箱容量、初始电量（可为单个数值或列表/字典）、
        能耗率、充电速率、行驶速度以及可选的载重容量。

    返回一个包含以下字段的字典：

    ``total_cost``
        该方案的总成本。
    ``remaining_energies``
        每辆车在完成各自路径后的剩余电量，可直接作为下一周期的初始电量。
    ``route_summaries``
        逐车的明细，包括距离成本、时间成本、可行性判定等。
    """

    evaluation = evaluate_routes_with_energy(
        routes,
        tasks_info,
        distance_matrix,
        tank_capacity,
        initial_energy,
        fuel_consumption_rate,
        charging_rate,
        velocity,
        load_capacity,
    )

    return {
        'total_cost': evaluation['total_cost'],
        'remaining_energies': evaluation['remaining_energies'],
        'route_summaries': evaluation['route_summaries'],
    }


def load_period_task_data(file: str) -> Dict[str, Any]:
    """读取实例文件并返回便于逐周期调用的数据结构。

    该函数在不修改原有解析逻辑的前提下复用 :func:`load_multi_period_instance`
    组装输出，方便直接拿到 ``tasks_info`` 与 ``distance_matrix``。返回的字典
    结构清晰，对接逐周期的手动验证流程更为便捷。
    """

    instance = load_multi_period_instance(file)

    period_summaries = [
        {
            'name': period.name,
            'nodes': period.nodes,
            'customers': period.customers,
            'tasks_info': period.tasks_info,
            'distance_matrix': period.distance_matrix,
        }
        for period in instance.periods
    ]

    return {
        'vehicle': instance.vehicle,
        'depot': instance.depot,
        'fuel_stations': instance.fuel_stations,
        'periods': period_summaries,
    }


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
# def list_problem_files(folder: str) -> List[str]:
#     """Return all txt instance files inside ``folder``."""
#     folder_path = Path(folder)
#     files = sorted(str(p) for p in folder_path.glob("*.txt"))
#     return files

# ---------------------------------------------------------------------------
# Dataclasses & typed containers used by the instance loading helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle configuration shared by all periods of an instance."""

    tank_capacity: float
    now_energy: Union[float, List[float]]
    load_capacity: float
    fuel_consumption_rate: float
    charging_rate: float
    velocity: float


@dataclass(frozen=True)
class PeriodData:
    """All information required to solve a single period."""

    name: str
    customers: List[List[Any]]
    nodes: List[str]
    tasks_info: Dict[str, Dict[str, Any]]
    distance_matrix: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class MultiPeriodInstance:
    """Representation of an instance that may contain multiple periods."""

    source: str
    vehicle: VehicleConfig
    depot: List[Any]
    fuel_stations: List[List[Any]]
    periods: List[PeriodData]


def list_problem_files(folder: str) -> List[str]:
    """
    返回指定文件夹中所有的 .txt 实例文件。
    文件列表会根据文件名中的数字进行正确的数值排序。
    例如：instance_1.txt, instance_2.txt, ..., instance_10.txt
    """
    folder_path = Path(folder)

    # 1. 获取所有 .txt 文件的 Path 对象列表
    # 使用 list() 预先收集所有文件，避免迭代器问题
    files_as_paths = list(folder_path.glob("*.txt"))

    # 2. 定义一个辅助函数，用于从路径对象中提取文件名中的数字
    def get_number_from_path(path_obj: Path) -> int:
        """从文件名（如 'Instance_10.txt'）中提取数字（如 10）"""
        # re.search(r'\d+', ...) 会在字符串中查找一个或多个连续的数字
        match = re.search(r'\d+', path_obj.name)

        # 如果找到了数字
        if match:
            # match.group(0) 返回找到的数字字符串（如 '10'）
            # int() 将其转换为整数（如 10）
            return int(match.group(0))

        # 如果文件名中没有数字，返回一个极大值，让它排在最后面，以防出错
        return float('inf')

    # 3. 使用这个辅助函数作为 sorted 的 key，对文件路径列表进行排序
    sorted_files_as_paths = sorted(files_as_paths, key=get_number_from_path)

    # 4. 将排序好的 Path 对象列表转换回字符串路径列表并返回
    return [str(p) for p in sorted_files_as_paths]


def _parse_energy_values(raw_value: str) -> Union[float, List[float]]:
    """Parse the energy configuration value which may contain multiple entries."""

    cleaned = raw_value.replace(';', ',')
    values = [float(value.strip()) for value in cleaned.split(',') if value.strip()]

    if not values:
        raise ValueError("No initial energy values provided in instance file.")

    if len(values) == 1:
        return values[0]

    return values

def _read_targets_block(file_obj) -> Tuple[List[Any], List[List[Any]], List[List[Any]]]:
    """Read the first block containing depot, stations and customer definitions."""

    target_line = file_obj.readline()

    customers: List[List[Any]] = []
    fuel_stations: List[List[Any]] = []
    depot: Optional[List[Any]] = None

    while target_line and target_line != '\n':
        stl = target_line.split()
        if len(stl) < 8:
            raise ValueError(f"Malformed target line: '{target_line.strip()}'")

        idx = int(stl[0][1:])
        new_target = [
            stl[0],
            idx,
            float(stl[2]),
            float(stl[3]),
            int(float(stl[4])),
            int(float(stl[5])),
            int(float(stl[6])),
            int(float(stl[7])),
        ]

        if stl[1] == 'd':
            depot = new_target
        elif stl[1] == 'f':
            fuel_stations.append(new_target)
        elif stl[1] == 'c':
            customers.append(new_target)
        else:
            raise ValueError(f"Unknown target type '{stl[1]}' in line '{target_line.strip()}'")

        target_line = file_obj.readline()

    if depot is None:
        raise ValueError("Instance file is missing a depot definition.")

    return depot, customers, fuel_stations


def _read_vehicle_configuration(file_obj) -> VehicleConfig:
    """Parse the configuration section that contains vehicle parameters."""

    configuration_line = file_obj.readline()
    if not configuration_line:
        raise ValueError("Unexpected end of file while reading vehicle configuration.")
    tank_capacity = float(configuration_line.split('/')[1])

    configuration_line = file_obj.readline()
    if not configuration_line:
        raise ValueError("Unexpected end of file while reading vehicle now energy.")
    now_energy_raw = configuration_line.split('/')[1]
    now_energy = _parse_energy_values(now_energy_raw)

    configuration_line = file_obj.readline()
    if not configuration_line:
        raise ValueError("Unexpected end of file while reading load capacity.")
    load_capacity = float(configuration_line.split('/')[1])

    configuration_line = file_obj.readline()
    if not configuration_line:
        raise ValueError("Unexpected end of file while reading fuel consumption rate.")
    fuel_consumption_rate = float(configuration_line.split('/')[1])

    configuration_line = file_obj.readline()
    if not configuration_line:
        raise ValueError("Unexpected end of file while reading charging rate.")
    charging_rate = float(configuration_line.split('/')[1])

    configuration_line = file_obj.readline()
    if not configuration_line:
        raise ValueError("Unexpected end of file while reading vehicle velocity.")
    velocity = float(configuration_line.split('/')[1])

    return VehicleConfig(
        tank_capacity=tank_capacity,
        now_energy=now_energy,
        load_capacity=load_capacity,
        fuel_consumption_rate=fuel_consumption_rate,
        charging_rate=charging_rate,
        velocity=velocity,
    )


def _parse_period_sections(raw_text: str) -> List[Tuple[Optional[str], List[List[Any]]]]:
    """Parse additional period blocks from the remaining part of the instance file."""

    if not raw_text:
        return []

    periods: List[Tuple[Optional[str], List[List[Any]]]] = []
    current_customers: List[List[Any]] = []
    current_name: Optional[str] = None

    def flush_period():
        nonlocal current_customers, current_name
        if current_customers:
            periods.append((current_name, current_customers))
            current_customers = []
            current_name = None

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        normalized = stripped.strip('[]').lstrip('#').strip()
        lower = normalized.lower()
        if lower.startswith('period'):
            flush_period()
            current_name = normalized or None
            continue

        stl = stripped.split()
        if len(stl) < 8:
            raise ValueError(f"Malformed customer definition in period block: '{stripped}'")

        if stl[1] != 'c':
            raise ValueError(
                "Only customer (type 'c') entries are allowed inside period-specific blocks."
            )

        idx = int(stl[0][1:])
        new_target = [
            stl[0],
            idx,
            float(stl[2]),
            float(stl[3]),
            int(float(stl[4])),
            int(float(stl[5])),
            int(float(stl[6])),
            int(float(stl[7])),
        ]
        current_customers.append(new_target)

    flush_period()
    return periods


def get_vehicle_energy(now_energy, route_idx: int, default: Optional[float] = None) -> float:
    """Return the initial energy for the vehicle assigned to ``route_idx``.

    ``now_energy`` can be a scalar, a list/tuple, a numpy array or a dictionary. If the
    requested index is not available, fall back to ``default`` when provided.
    """

    if isinstance(now_energy, dict):
        if route_idx in now_energy:
            return now_energy[route_idx]
        if "default" in now_energy:
            return now_energy["default"]
        if now_energy:
            return next(iter(now_energy.values()))

    if isinstance(now_energy, (list, tuple, np.ndarray)):
        if route_idx < len(now_energy):
            return now_energy[route_idx]
        return 1500.0

    if now_energy is None:
        if default is None:
            raise ValueError(f"No energy value available for vehicle index {route_idx}.")
        return default

    return float(now_energy)

def _coerce_energy_sequence(
        now_energy: Union[float, Sequence[float], np.ndarray, Dict[Any, Any], None],
        fallback: float,
) -> Tuple[List[float], float]:
    """Return a list representation of ``now_energy`` and the fallback value."""

    if isinstance(now_energy, dict):
        default_value = float(now_energy.get("default", fallback))
        numeric_keys = [key for key in now_energy.keys() if isinstance(key, int)]
        if numeric_keys:
            length = max(numeric_keys) + 1
            values = [default_value] * length
            for key in numeric_keys:
                values[key] = float(now_energy[key])
        else:
            values = []
        return values, default_value

    if isinstance(now_energy, (list, tuple, np.ndarray)):
        values = [float(value) for value in now_energy]
        return values, fallback

    if now_energy is None:
        return [], fallback

    value = float(now_energy)
    return [value], fallback


def compute_remaining_energies(
        routes: Sequence[Sequence[Any]],
        tasks_info,
        distance_matrix,
        tank_capacity,
        now_energy,
        fuel_consumption_rate,
        charging_rate,
        velocity,
) -> List[float]:
    """Return the remaining energy for each vehicle after completing its route."""

    base_energies, fallback_value = _coerce_energy_sequence(now_energy, tank_capacity)
    energies: List[float] = list(base_energies)

    # Ensure the result can hold all routes that were actually used.
    if len(energies) < len(routes):
        energies.extend([fallback_value] * (len(routes) - len(energies)))

    for idx, route in enumerate(routes):
        vehicle_energy = get_vehicle_energy(now_energy, idx, tank_capacity)
        remaining = calculate_remaining_tank(
            route,
            tasks_info,
            distance_matrix,
            tank_capacity,
            vehicle_energy,
            fuel_consumption_rate,
        )
        if idx < len(energies):
            energies[idx] = float(remaining)
        else:
            energies.append(float(remaining))

    return energies


def load_problem_instance(file):
    with open(file) as f:
        f.readline()  # ignore header

        depot, customers, fuel_stations = _read_targets_block(f)
        vehicle_config = _read_vehicle_configuration(f)

        nodes, tasks_info = build_tasks_info(depot, customers, fuel_stations)
        distance_matrix = compute_dist_matrix(nodes, tasks_info)

        return (
            vehicle_config.tank_capacity,
            vehicle_config.now_energy,
            vehicle_config.load_capacity,
            vehicle_config.fuel_consumption_rate,
            vehicle_config.charging_rate,
            vehicle_config.velocity,
            depot,
            customers,
            fuel_stations,
            nodes,
            tasks_info,
            distance_matrix,
        )


def load_multi_period_instance(file: str) -> MultiPeriodInstance:
    """Load an instance that may contain multiple customer-period blocks."""

    with open(file) as f:
        f.readline()  # ignore header
        depot, base_customers, fuel_stations = _read_targets_block(f)
        vehicle_config = _read_vehicle_configuration(f)
        remaining = f.read()

    periods: List[PeriodData] = []

    def make_period(customers: List[List[Any]], name: str) -> None:
        nodes, tasks_info = build_tasks_info(depot, customers, fuel_stations)
        distance_matrix = compute_dist_matrix(nodes, tasks_info)
        periods.append(
            PeriodData(
                name=name,
                customers=customers,
                nodes=nodes,
                tasks_info=tasks_info,
                distance_matrix=distance_matrix,
            )
        )

    if base_customers:
        make_period(base_customers, "period_1")

    parsed_additional_periods = _parse_period_sections(remaining)
    next_index = len(periods) + 1
    for maybe_name, customers in parsed_additional_periods:
        period_name = maybe_name or f"period_{next_index}"
        make_period(customers, period_name)
        next_index += 1

    # If no customers were found at all, create a placeholder empty period so that
    # downstream code still receives a valid structure.
    if not periods:
        make_period([], "period_1")

    return MultiPeriodInstance(
        source=str(file),
        vehicle=vehicle_config,
        depot=depot,
        fuel_stations=fuel_stations,
        periods=periods,
    )

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

        if tasks_info[route[i]]['Type'] == 'f':
            missing_energy = tank_capacity - calculate_remaining_tank(route, tasks_info, distance_matrix,
                                                                      tank_capacity, now_energy,
                                                                      fuel_consumption_rate, until=route[i])
            tasks_info[route[i]]['Service-Time'] = missing_energy * charging_rate

        elapsed_time += tasks_info[route[i]]['Service-Time']

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

            if now_energy < 0:
                return True

            if tasks_info[t]['Type'] == 'f':
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

            # 记录当前节点的实际到达时间
            arrival_times.append(elapsed_time)

            if tasks_info[i]['Type'] == 'f':

                missing_energy = tank_capacity - calculate_remaining_tank(route, tasks_info, distance_matrix,
                                                                          tank_capacity, now_energy,
                                                                          fuel_consumption_rate, until=i)

                tasks_info[i]['Service-Time'] = missing_energy * charging_rate

            elapsed_time += tasks_info[i]['Service-Time']

        last = i
    return arrival_times


def calculate_remaining_tank1(route, tasks_info, distance_matrix, tank_capacity, now_energy, fuel_consumption_rate,
                             until=None):

    last = None
    # now_energy = float(now_energy)
    now_energy = now_energy

    total_consumption = 0
    for t in route:

        if last is not None:
            distance = distance_matrix[last][t]
            consumption = distance * fuel_consumption_rate
            # total_consumption += consumption
            now_energy -= consumption

            if until == t:
                return now_energy

            if tasks_info[t]['Type'] == 'f':
                now_energy = tank_capacity

        last = t
        # print(total_consumption)
    return now_energy


def calculate_remaining_tank(route, tasks_info, distance_matrix, tank_capacity, now_energy, fuel_consumption_rate,
                             until=None):
    last = None
    # now_energy = float(now_energy)
    now_energy = now_energy
    total_consumption = 0
    for t in route:
        if last is not None:
            distance = distance_matrix[last][t]
            consumption = distance * fuel_consumption_rate
            # total_consumption += consumption
            now_energy -= consumption

            if until == t:
                return now_energy

            if tasks_info[t]['Type'] == 'f':
                now_energy = tank_capacity

        last = t
        # print(total_consumption)
    return now_energy


def calculate_demand(route, tasks_info, distance_matrix, tank_capacity, now_energy, fuel_consumption_rate,
                     charging_rate, velocity):
    arrival_times = calculate_arrival_times(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity)
    demand = []

    for i in range(len(route)):
        if tasks_info[route[i]]['Type'] == 'f':
            tasks_demand = 0
            demand.append(tasks_demand)
        elif tasks_info[route[i]]['Type'] == 'c':
            tasks_demand = (48 - tasks_info[route[i]]['stock-at-call-time'] + (arrival_times[i] -
                                                                             tasks_info[route[i]]['call-time']) / 30) * 0.75
            demand.append(tasks_demand)
        elif tasks_info[route[i]]['Type'] == 'd':
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
    dist = calculate_route_distance(route, distance_matrix)
    # print(dist)
    return now_energy - dist * fuel_consumption_rate < 0.2 * tank_capacity


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

        if tasks_info[route[i]]['Type'] != 'f':
            reachable_stations = get_reachable_charging_stations(route[:i], nodes, tasks_info, distance_matrix,
                                                                 tank_capacity, now_energy, fuel_consumption_rate,
                                                                 until=route[i])
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


def make_route_feasible(route, nodes, tasks_info, distance_matrix, tank_capacity, now_energy, vehicle_energy,
                                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    reachable_stations = get_reachable_charging_stations(route, nodes, tasks_info, distance_matrix,
                                                         tank_capacity, now_energy, fuel_consumption_rate,
                                                         until=route[-1])
    if reachable_stations is None:
        return None
    nearest_station = min(
        reachable_stations,
        key=lambda candidate: distance_matrix[route[-1]][candidate]
    )
    feasible_route = route.insert(-1, nearest_station)

    if not is_feasible(feasible_route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
        # print("NO")
        return None
    else:
        # print("YES")
        # print(best_feasible_route)
        return feasible_route


    # if need_charge(route, distance_matrix, vehicle_energy, fuel_consumption_rate, tank_capacity):
    #     while is_feasible(route, tasks_info, distance_matrix, tank_capacity, now_energy,
    #                         fuel_consumption_rate, charging_rate, velocity, load_capacity):
    #         reachable_stations = get_reachable_charging_stations(route, nodes, tasks_info, distance_matrix,
    #                                                          tank_capacity, now_energy, fuel_consumption_rate,
    #                                                          until=route[-1])#如果没有可以找到的，也要移除客户



def process_route(state, nodes, tasks_info, distance_matrix, tank_capacity, now_energy,
                  fuel_consumption_rate, charging_rate, velocity, load_capacity):
    unvisited_customers = []

    for idx, value in enumerate(state.routes):
        vehicle_energy = get_vehicle_energy(now_energy, idx, tank_capacity)
        # 判断当前路径是否需要充电
        while need_charge(state.routes[idx], distance_matrix, vehicle_energy, fuel_consumption_rate, tank_capacity):

            k = make_route_feasible_and_best(value, nodes, tasks_info, distance_matrix, tank_capacity, vehicle_energy,
                                             fuel_consumption_rate, charging_rate, velocity, load_capacity)
            if k is not None:
                # 找到可行的插入点，更新路径
                state.routes[idx] = k
                break  # 找到充电位置，跳出while循环，继续下一条路径
            else:
                # 找不到插入点，移除最小到达时间的客户点
                earliest_customer = find_earliest_customer(state.routes[idx], tasks_info)
                state.routes[idx].remove(earliest_customer)    # 从当前路径中移除
                unvisited_customers.append(earliest_customer)  # 将该客户点标记为未访问
                # 路径调整后，继续判断是否需要充电
    while unvisited_customers:
        # print("Current unvisited_customers:", unvisited_customers)

        # 使用k-最近邻域启发式为未访问的客户生成新的路径
        new_routes = innh(unvisited_customers, tasks_info, distance_matrix, tank_capacity, now_energy,
                          fuel_consumption_rate, charging_rate, velocity, load_capacity, k=3,
                          start_idx=len(state.routes))
        unvisited_customers = []
        # print(new_routes)
        # 对新的路径进行充电站插入操作
        for idx, value in enumerate(new_routes):
            route_idx = len(state.routes) + idx
            vehicle_energy = get_vehicle_energy(now_energy, route_idx, tank_capacity)
            # print(value)
            # print(self.need_charge(value))
            while need_charge(value, distance_matrix, vehicle_energy, fuel_consumption_rate, tank_capacity):
                k = make_route_feasible_and_best(value, nodes, tasks_info, distance_matrix, tank_capacity,
                                                 vehicle_energy,
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
        state.routes.extend(new_routes)
    # print("22222222222222")
    return state


def innh(customers, tasks_info, distance_matrix, tank_capacity, now_energy,
         fuel_consumption_rate, charging_rate, velocity, load_capacity, k=3, start_idx=0):
    giant_route = []
    serviced_customers = set()
    unserved_customers = customers.copy()

    while unserved_customers:
        route_idx = start_idx + len(giant_route)
        vehicle_energy = get_vehicle_energy(now_energy, route_idx, tank_capacity)
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
            if not is_nc_feasible(route, tasks_info, distance_matrix, tank_capacity, vehicle_energy,
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
        if tasks_info[v]['Type'] == 'c':
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
            if tasks_info[route[i]]['Type'] == 'f':
                time_cost += 0
            elif tasks_info[route[i]]['Type'] == 'c':
                time_cost += tasks_info[route[i]]['Due-Date']-arrival_times[i]
            # print(time_cost)
        route_cost = dist_cost + 0.5 * time_cost + 200

    return route_cost


def calculate_nc_route_cost(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                            fuel_consumption_rate, charging_rate, velocity, load_capacity):
    dist_cost = calculate_route_distance(route, distance_matrix)
    # print(dist_cost)
    arrival_times = calculate_arrival_times(route, tasks_info, distance_matrix, tank_capacity, now_energy,
                                            fuel_consumption_rate, charging_rate, velocity)
    time_cost = 0
    for i in range(1, len(route)):
        # print(route[i])
        if tasks_info[route[i]]['Type'] == 'c':
            time_cost += tasks_info[route[i]]['Due-Date']-arrival_times[i]
        # print(time_cost)
        elif tasks_info[route[i]]['Type'] == 'f':
            time_cost += 0
    route_cost = dist_cost + 0.5 * time_cost

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
