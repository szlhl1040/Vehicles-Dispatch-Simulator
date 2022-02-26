from dataclasses import dataclass
from typing import List, Mapping, Tuple

import pandas as pd

from domain.local_region_bound import LocalRegionBound


@dataclass
class Node:
    id: int
    longitude: float
    latitude: float


class NodeManager:
    def __init__(self, node_df: pd.DataFrame):
        self.__node_list: List[Node] = [
            Node(
                id=row["ID"],
                longitude=round(row["longitude"], 7),
                latitude=round(row["latitude"], 7)
            )
            for row in node_df.iterrows()
        ]

    @property
    def node_id_list(self) -> List[int]:
        return [node.id for node in self.__node_list]

    @property
    def node_location(self) -> List[List[float]]:
        return [[node.longitude, node.latitude] for node in self.__node_list] 

    def restrict_area(self, local_region_bound: LocalRegionBound) -> None:
        tmp_node_list = []
        for node in self.__node_list:
            if self.__is_node_in_limit_region(local_region_bound, node):
                tmp_node_list.append(node)
        self.__node_list = tmp_node_list

    def __len__(self):
        return len(self.__node_list)

    @classmethod
    def __is_node_in_limit_region(cls, local_region_bound: LocalRegionBound, node: Node) -> bool:
        if (
            node.longitude < local_region_bound.west_bound
            or node.longitude > local_region_bound.east_bound
        ):
            return False
        elif (
            node.latitude < local_region_bound.south_bound
            or node.latitude > local_region_bound.north_bound
        ):
            return False

        return True