from dataclasses import dataclass

import numpy as np
import pandas as pd

from domain.local_region_bound import LocalRegionBound


@dataclass
class Node:
    id: int
    longitude: float
    latitude: float


class NodeManager:
    def __init__(self, node_df: pd.DataFrame) -> None:
        self.__node_list = [
            Node(
                id=row["NodeID"],
                longitude=row["Longitude"],
                latitude=row["Latitude"],
            )
            for _, row in node_df.iterrows()
        ]

    @property
    def node_locations(self) -> np.ndarray:
        return np.array([[round(node.longitude, 7), round(node.latitude, 7)] for node in self.__node_list])

    @property
    def node_id_list(self) -> np.ndarray:
        return np.array([node.id for node in self.__node_list])

    def restrict_nodes(self, local_region_bound: LocalRegionBound) -> None:
        tmp_node_list = []
        for node in self.__node_list:
            if self.__is_node_in_limit_region(node, local_region_bound):
                tmp_node_list.append(node)
        self.__node_list = tmp_node_list

    @classmethod
    def __is_node_in_limit_region(cls, node: Node, local_region_bound: LocalRegionBound) -> bool:
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

    def __len__(self) -> int:
        return len(self.__node_list)