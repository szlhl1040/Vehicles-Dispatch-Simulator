import random

from objects import Area, Vehicle


class DispatchModuleInterface:
    def __call__(self, vehicle: Vehicle) -> bool:
        raise NotImplementedError


class RandomDispatch(DispatchModuleInterface):
    def __call__(self, vehicle: Vehicle) -> bool:
        current_area: Area = vehicle.area
        candidate_area = current_area.neighbor + [current_area]
        next_area = random.choice(candidate_area)
        vehicle.area = next_area

        return current_area == next_area