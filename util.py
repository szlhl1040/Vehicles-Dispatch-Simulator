from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import numpy as np


def haversine(lon1, lat1, lon2, lat2) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


class DataModule:
    def __init__(self):
        self.__date = datetime(2016, 6, 1)

    @property
    def date(self) -> str:
        return f"{self.__date.month}".zfill(2) + f"{self.__date.day}".zfill(2)

    def next(self) -> bool:
        next_day = self.__date + timedelta(days=1)
        if next_day.month != 6:
            False
        self.__date = next_day
        return True

    def __str__(self) -> str:
        return self.date
