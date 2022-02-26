from typing import Any, List

import numpy as np
import pandas as pd


class Node:
    def __init__(self, id: int, longitude: float, latitude: float):
        self.id: int = id
        self.longitude: float = longitude
        self.latitude: float = latitude

    def __eq__(self, other: Any):
        if isinstance(other) == Node:
            return False
        if other.id != self.id:
            return False
        return True


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
        self.__node_index = {node.id: idx for idx, node in enumerate(self.__node_list)}
        self.__node_dict = {node.id: node for node in self.__node_list}

    @property
    def node_locations(self) -> np.ndarray:
        return np.array(
            [
                [round(node.longitude, 7), round(node.latitude, 7)]
                for node in self.__node_list
            ]
        )

    @property
    def node_id_list(self) -> np.ndarray:
        return np.array([node.id for node in self.__node_list])

    def get_nodes(self) -> List[Node]:
        return [node for node in self.__node_list]

    def get_node_index(self, node_id: int) -> int:
        return self.__node_index[node_id]

    def get_node(self, node_id) -> Node:
        return self.__node_dict[node_id]

    def __len__(self) -> int:
        return len(self.__node_list)
