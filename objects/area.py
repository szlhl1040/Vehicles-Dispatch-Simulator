from abc import abstractmethod
from datetime import datetime
from typing import List, Mapping, Tuple

from objects.node import Node
from objects.order import Order
from objects.vehicle import Vehicle


class Area:
    def __init__(
        self,
        id,
        nodes: List[Node],  # node_index: {longitude, latitude}
        neighbor: List["Area"],
        rebalance_number,
        idle_vehicles,
        vehicles_arrive_time: Mapping[Vehicle, datetime],
        orders,
    ):
        self.id = id
        self.nodes: List[Node] = nodes
        self.neighbor: List[Area] = neighbor
        self.rebalance_number = rebalance_number
        self.idle_vehicles: List[Vehicle] = idle_vehicles
        self.vehicles_arrive_time = vehicles_arrive_time
        self.orders: List[Order] = orders
        self.per_rebalance_idle_vehicles = 0
        self.later_rebalance_idle_vehicles = 0
        self.per_match_idle_vehicles = 0
        self.rebalance_frequency = 0
        self.dispatch_number = 0
        self.per_dispatch_idle_vehicles = 0
        self.later_dispatch_idle_vehicles = 0

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def arrive_cluster_update(self, vehicle: Vehicle):
        self.idle_vehicles.append(vehicle)
        self.vehicles_arrive_time.pop(vehicle)

    @abstractmethod
    def example(self):
        raise NotImplementedError


class Cluster(Area):
    def reset(self):
        self.rebalance_number = 0
        self.idle_vehicles.clear()
        self.vehicles_arrive_time.clear()
        self.orders.clear()
        self.per_rebalance_idle_vehicles = 0
        self.per_match_idle_vehicles = 0

    def example(self):
        print("Order Example output")
        print("ID:", self.id)
        print("Nodes:", self.nodes)
        print("Neighbor:", self.neighbor)
        print("RebalanceNumber:", self.rebalance_number)
        print("IdleVehicles:", self.idle_vehicles)
        print("VehiclesArrivetime:", self.vehicles_arrive_time)
        print("Orders:", self.orders)


class Grid(Area):
    def reset(self):
        self.rebalance_number = 0
        self.idle_vehicles.clear()
        self.vehicles_arrive_time.clear()
        self.orders.clear()
        self.per_rebalance_idle_vehicles = 0
        self.per_match_idle_vehicles = 0

    def example(self):
        print("ID:", self.id)
        print("Nodes:", self.nodes)
        print("Neighbor:[", end=" ")
        for i in self.neighbor:
            print(i.id, end=" ")
        print("]")
        print("RebalanceNumber:", self.rebalance_number)
        print("IdleVehicles:", self.idle_vehicles)
        print("VehiclesArrivetime:", self.vehicles_arrive_time)
        print("Orders:", self.orders)
        print()
