import math
import random
from typing import List, Dict, Any, Tuple


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
        'kind': 'depot',
        'x': depot[2],
        'y': depot[3],
        'tw_start': depot[4],
        'tw_end': depot[5],
        'demand': depot[6],
        'battery': depot[7],
    }

    # customers
    for c in customers:
        tasks_info[c[0]] = {
            'kind': 'customer',
            'x': c[2],
            'y': c[3],
            'tw_start': c[4],
            'tw_end': c[5],
            'demand': c[6],
            'battery': c[7],
        }

    # fuel stations
    for s in fuel_stations:
        tasks_info[s[0]] = {
            'kind': 'station',
            'x': s[2],
            'y': s[3],
            'tw_start': s[4],
            'tw_end': s[5],
            'demand': s[6],
            'battery': s[7],
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

def unvisitied(a):
    flattened_new_routes = [customer for route in a for customer in route]

def innh(customers):
    serviced = set()
    customers = [c for c in customers if c not in serviced]  # 重新赋值
    customers.remove(2)  # 直接操作列表
    serviced.add(1)

# —— 使用示例 —— #

if __name__ == '__main__':
    depot = ['D0', 'd', 10.0, 20.0, 8, 18, 0, 100]
    customers = [['C1', 'c', 12.0, 22.0, 9, 12, 10, 100],
                 ['C2', 'c', 15.0, 25.0, 10, 14, 5, 100]]
    fuel_stations = [['S1', 'f', 11.0, 21.0, 0, 24, 0, 100],
                     ['S2', 'f', 13.0, 23.0, 0, 24, 0, 100]]

    nodes, tasks_info = build_tasks_info(depot, customers, fuel_stations)
    dist_matrix = compute_dist_matrix(nodes, tasks_info)


    unvisited_customers = [1,2,3,4,5,1,1,2]
    a = [[2,3,4,5],[2]]
    flattened_new_routes = [customer for route in a for customer in route]
    unvisited_customers = [customer for customer in unvisited_customers if customer not in flattened_new_routes]
    route = ['D0', 'C1', 'C4', 'C3', 'C2', 'C5', 'S1', 'C6', 'C7', 'D0']
    print(route[1:-1])
    print(route[1:-2])
    # 验证
    print("节点顺序:", nodes)
    print("C1 属性:", tasks_info['C1'])
    print("C1→S2 距离:", dist_matrix['C1']['C2'])

    a = {}
    a[5] = 1
    a[1] = 2
    a[3] = 5
    chosen = sorted(a, reverse=False)
    print(chosen)


    print(random.sample(nodes, 3))

