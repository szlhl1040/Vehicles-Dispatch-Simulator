from dataclasses import dataclass

@dataclass(frozen=True)
class LocalRegionBound:
    west_bound: float
    east_bound: float
    north_bound: float
    south_bound: float

    def __str__(self):
        return "({self.west_bound}, {self.east_bound}, {self.south_bound}, {self.north_bound})"
