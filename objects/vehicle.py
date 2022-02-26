from typing import List

from objects.order import Order


class Vehicle(object):
    def __init__(
        self,
        id: int,
        location_node_id: int,
        area,
        orders: List[Order],
        delivery_node_id: int,
    ):
        self.id = id  # This vehicle's ID
        self.location_node_id = location_node_id  # Current vehicle's location
        self.area = area  # Which cluster the current vehicle belongs to
        self.orders: List[Order] = orders  # Orders currently on board
        self.delivery_node_id = delivery_node_id  # Next destination of current vehicle

    def arrive_vehicle_update(self, delivery_area) -> None:
        self.location_node_id = self.delivery_node_id
        self.delivery_node_id = None
        self.area = delivery_area
        if len(self.orders):
            self.orders.clear()

    def reset(self) -> None:
        self.orders.clear()
        self.delivery_node_id = None

    def example(self) -> None:
        print("Vehicle Example output")
        print("ID:", self.id)
        print("LocationNode:", self.location_node_id)
        print("Area:", self.area)
        print("Orders:", self.orders)
        print("DeliveryPoint:", self.delivery_node_id)
        print()
