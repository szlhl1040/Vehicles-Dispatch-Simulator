from typing import List

from objects.order import Order


class Vehicle(object):    
    def __init__(
        self,
        id: int,
        location_node,
        area,
        orders: List[Order],
        delivery_point
    ):
        self.id = id                                            #This vehicle's ID    
        self.location_node = location_node                        #Current vehicle's location
        self.area = area                                  #Which cluster the current vehicle belongs to
        self.orders: List[Order] = orders                                    #Orders currently on board
        self.delivery_point = delivery_point                      #Next destination of current vehicle

    def arrive_vehicle_update(self, delivery_area) -> None:
        self.location_node = self.delivery_point
        self.delivery_point = None
        self.area = delivery_area
        if len(self.orders):
            self.orders.clear()

    def reset(self) -> None:
        self.orders.clear()
        self.delivery_point = None

    def example(self) -> None:
        print("Vehicle Example output")
        print("ID:",self.id)
        print("LocationNode:",self.location_node)
        print("Area:",self.area)
        print("Orders:",self.orders)
        print("DeliveryPoint:",self.delivery_point)
        print()