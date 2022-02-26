import datetime
from dataclasses import dataclass


@dataclass
class StaticsService:
    def __init__(self):
        self.__order_num: int = 0
        self.__reject_num: int = 0
        self.__dispatch_num: int = 0
        self.__totally_dispatch_cost: int = 0
        self.__totally_wait_time: int = 0
        self.__totally_update_time = datetime.timedelta()
        self.__totally_reward_time = datetime.timedelta()
        self.__totally_next_state_time = datetime.timedelta()
        self.__totally_learning_time = datetime.timedelta()
        self.__totally_dispatch_time = datetime.timedelta()
        self.__totally_match_time = datetime.timedelta()
        self.__totally_demand_predict_time = datetime.timedelta()

    def reset(self):
        self.__order_num: int = 0
        self.__reject_num: int = 0
        self.__dispatch_num: int = 0
        self.__totally_dispatch_cost: int = 0
        self.__totally_wait_time: int = 0
        self.__totally_update_time = datetime.timedelta()
        self.__totally_reward_time = datetime.timedelta()
        self.__totally_next_state_time = datetime.timedelta()
        self.__totally_learning_time = datetime.timedelta()
        self.__totally_dispatch_time = datetime.timedelta()
        self.__totally_match_time = datetime.timedelta()
        self.__totally_demand_predict_time = datetime.timedelta()

    def increment_order_num(self) -> None:
        self.__order_num += 1

    @property
    def reject_num(self) -> int:
        return self.__reject_num

    def increment_reject_num(self) -> None:
        self.__reject_num += 1

    @property
    def dispatch_num(self) -> int:
        return self.__dispatch_num

    def increment_dispatch_num(self) -> None:
        self.__dispatch_num += 1

    @property
    def totally_dispatch_cost(self) -> int:
        return self.__totally_dispatch_cost

    def add_dispatch_cost(self, dispatch_cost: int) -> None:
        self.__totally_dispatch_cost += dispatch_cost

    @property
    def totally_update_time(self) -> datetime.timedelta:
        return self.__totally_update_time

    def add_totally_update_time(self, delta: datetime.timedelta) -> None:
        self.__totally_update_time += delta

    @property
    def totally_match_time(self) -> datetime.timedelta:
        return self.__totally_match_time

    def add_totally_match_time(self, delta: datetime.timedelta) -> None:
        self.__totally_match_time += delta

    @property
    def totally_reward_time(self) -> datetime.timedelta:
        return self.__totally_reward_time

    def add_totally_reward_time(self, delta: datetime.timedelta) -> None:
        self.__totally_reward_time += delta

    @property
    def totally_next_state_time(self) -> datetime.timedelta:
        return self.__totally_next_state_time

    def add_totally_next_state_time(self, delta: datetime.timedelta) -> None:
        self.__totally_next_state_time += delta

    @property
    def totally_learning_time(self) -> datetime.timedelta:
        return self.__totally_learning_time

    def add_totally_learning_time(self, delta: datetime.timedelta) -> None:
        self.__totally_learning_time += delta

    @property
    def totally_demand_predict_time(self) -> datetime.timedelta:
        return self.__totally_demand_predict_time

    def add_totally_demand_predict_time(self, delta: datetime.timedelta) -> None:
        self.__totally_demand_predict_time += delta

    @property
    def totally_dispatch_time(self) -> datetime.timedelta:
        return self.__totally_dispatch_time

    def add_totally_dispatch_time(self, delta: datetime.timedelta) -> None:
        self.__totally_dispatch_time += delta

    @property
    def totally_wait_time(self) -> int:
        return self.__totally_wait_time

    def add_totally_wait_time(self, cost: int) -> None:
        self.__totally_wait_time += cost
