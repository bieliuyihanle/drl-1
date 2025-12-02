import sys
sys.path.insert(0, 'C:/Users/10133/Desktop/DR-ALNS-master/DR-ALNS/code/src')  # 替换为你的本地项目路径
from typing import List, Optional, Sequence, Union, TYPE_CHECKING

from routing.cvrp.alns_cvrp import cvrp_helper_functions

if TYPE_CHECKING:
    from routing.cvrp.alns_cvrp.cvrp_helper_functions import MultiPeriodInstance, PeriodData

# def evaluate_solution(routes, tasks_info, distance_matrix, tank_capacity, now_energy,
#                       fuel_consumption_rate, charging_rate, velocity):
#     total_cost = 0
#
#     for idx, route in enumerate(routes):
#         vehicle_energy = cvrp_helper_functions.get_vehicle_energy(now_energy, idx, tank_capacity)
#         dist_cost = cvrp_helper_functions.calculate_route_distance(route, distance_matrix)
#         # print(dist_cost)
#         arrival_times = cvrp_helper_functions.calculate_arrival_times(
#             route, tasks_info, distance_matrix, tank_capacity, vehicle_energy,
#             fuel_consumption_rate, charging_rate, velocity
#         )
#         time_cost = 0
#         for i in range(1, len(route)):
#             # print(route[i])
#             if tasks_info[route[i]]['Type'] == 'f':
#                 time_cost += 0
#             elif tasks_info[route[i]]['Type'] == 'c':
#                 time_cost += tasks_info[route[i]]['Due-Date'] - arrival_times[i]
#             # print(time_cost)
#         route_cost = dist_cost + 0.1 * time_cost + 1000
#         total_cost += route_cost
#
#     # can be removed in deployment, just for testing
#     # for route in routes:
#     #     if compute_route_load(route, demands_data) > truck_capacity:
#     #         print('TOO MUCH LOAD FOR TRUCK')
#     return total_cost

def evaluate_solution(routes, tasks_info, distance_matrix, tank_capacity, now_energy,
                      fuel_consumption_rate, charging_rate, velocity):
    evaluation = cvrp_helper_functions.evaluate_solution_cost_and_energy(
        routes,
        tasks_info=tasks_info,
        distance_matrix=distance_matrix,
        tank_capacity=tank_capacity,
        initial_energy=now_energy,
        fuel_consumption_rate=fuel_consumption_rate,
        charging_rate=charging_rate,
        velocity=velocity,
    )
    return evaluation['total_cost']

class cvrpEnv:
    def __init__(self, initial_solution, tank_capacity, now_energy, load_capacity, fuel_consumption_rate,
         charging_rate, velocity, depot, customers, fuel_stations,
         nodes, tasks_info, distance_matrix, problem_instance, seed, unassigned=None):
        self.customers = customers
        self.tasks_info = tasks_info
        self.distance_matrix = distance_matrix
        self.tank_capacity = tank_capacity
        self.now_energy = now_energy
        self.fuel_consumption_rate = fuel_consumption_rate
        self.charging_rate = charging_rate
        self.velocity = velocity
        self.load_capacity = load_capacity
        self.fuel_stations = fuel_stations
        self.nodes = nodes
        self.unassigned = unassigned if unassigned is not None else []

        self.seed = seed
        self.problem_instance = problem_instance

        self.routes = initial_solution


    @classmethod
    def from_period(
            cls,
            period: "PeriodData",
            instance: "MultiPeriodInstance",
            now_energy: Union[float, Sequence[float]],
            seed: int,
            initial_solution: Optional[Sequence[Sequence[str]]] = None,
            unassigned: Optional[Sequence[str]] = None,
    ) -> "cvrpEnv":
        """Instantiate ``cvrpEnv`` for a specific period of a multi-period instance."""

        vehicle = instance.vehicle
        problem_id = f"{instance.source}|{period.name}"
        return cls(
            list(initial_solution) if initial_solution is not None else [],
            vehicle.tank_capacity,
            now_energy,
            vehicle.load_capacity,
            vehicle.fuel_consumption_rate,
            vehicle.charging_rate,
            vehicle.velocity,
            instance.depot,
            period.customers,
            instance.fuel_stations,
            period.nodes,
            period.tasks_info,
            period.distance_matrix,
            problem_id,
            seed,
            unassigned=list(unassigned) if unassigned is not None else None,
        )

    def get_vehicle_energy(self, route_idx: int) -> float:
        return cvrp_helper_functions.get_vehicle_energy(self.now_energy, route_idx, self.tank_capacity)

    def remaining_energies(self) -> List[float]:
        return cvrp_helper_functions.compute_remaining_energies(
            self.routes,
            self.tasks_info,
            self.distance_matrix,
            self.tank_capacity,
            self.now_energy,
            self.fuel_consumption_rate,
            self.charging_rate,
            self.velocity,
        )

    def objective(self, best=False):
        score = evaluate_solution(self.routes, self.tasks_info, self.distance_matrix, self.tank_capacity, self.now_energy,
                                  self.fuel_consumption_rate, self.charging_rate, self.velocity)
        return score
