from datetime import datetime
from domain.arrive_info import ArriveInfo


class Order(object):
    def __init__(
        self,
        id: int,
        release_time: datetime,
        pick_up_node_id: int,
        delivery_node_id: int,
        pick_up_time_window,
        pick_up_wait_time,
        arrive_info: ArriveInfo,
        order_value: int,
    ):
        self.id: int = id  # This order's ID
        self.release_time: datetime = release_time  # Start time of this order
        self.pick_up_node_id: int = (
            pick_up_node_id  # The starting position of this order
        )
        self.delivery_node_id: int = delivery_node_id  # Destination of this order
        self.pick_up_time_window = (
            pick_up_time_window  # Limit of waiting time for this order
        )
        self.pick_up_wait_time = pick_up_wait_time  # This order's real waiting time from running in the simulator
        self.arrive_info: ArriveInfo = (
            arrive_info  # Processing information for this order
        )
        self.order_value = order_value  # The value of this order

    def arrive_order_time_record(self, arrive_time) -> None:
        self.arrive_info = "ArriveTime:" + str(arrive_time)

    def example(self) -> None:
        print("Order Example output")
        print("ID:", self.id)
        print("ReleasTime:", self.release_time)
        print("PickupPoint:", self.pick_up_node_id)
        print("DeliveryPoint:", self.delivery_node_id)
        print("PickupTimeWindow:", self.pick_up_time_window)
        print("PickupWaitTime:", self.pick_up_wait_time)
        print("ArriveInfo:", self.arrive_info)
        print()

    def reset(self) -> None:
        self.pick_up_wait_time = None
        self.arrive_info = None
