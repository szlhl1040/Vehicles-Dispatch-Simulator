import datetime
import os
import random
from pathlib import Path
from typing import List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import MINUTES, PICKUPTIMEWINDOW
from domain import (
    AreaMode,
    ArriveInfo,
    DemandPredictionMode,
    DispatchMode,
    LocalRegionBound,
)
from modules import DispatchModuleInterface, StaticsService
from modules.dispatch import RandomDispatch
from objects import Area, Cluster, Grid, NodeManager, Order, Vehicle
from preprocessing.readfiles import read_all_files, read_map, read_order
from util import haversine

###########################################################################

DATA_PATH = "./data/Order/modified"
TRAIN = "train"
TEST = "test"

base_data_path = Path(DATA_PATH)


class Simulation(object):
    """
    This simulator is used to simulate urban vehicle traffic.The system divides the day into several time slots.
    System information is updated at the beginning of each time slot to update vehicle arrivals and order completion.
    Then the system generates the order that needs to be started within the current time slot, and then finds the optimal
    idle vehicle to match the order. If the match fails or the recent vehicles have timed out, the order is marked as Reject.
    If it is successful, the vehicle service order is arranged. The shortest path in the road network first reaches the
    place where the order occurred, and then arrives at the order destination, and repeats matching the order until all
    the orders in the current time slot have been completed. Then the system generates orders that occur within the current
    time slot, finds the nearest idle vehicle to match the order, and if there is no idle vehicle or the nearest idle vehicle
    reaches the current position of the order and exceeds the limit time, the match fails, and if the match is successful, the
    selected vehicle service is arranged Order. After the match is successful, the vehicle's idle record in the current cluster
    is deleted, and the time to be reached is added to the cluster where the order destination is located. The vehicle must
    first arrive at the place where the order occurred, pick up the passengers, and then complete the order at the order destination.
    Repeat the matching order until a match All orders in this phase are completed.
    At the end of the matching phase, you can useyour own matching method to dispatch idle vehicles in each cluster to other
    clusters that require more vehicles to meet future order requirements.
    """

    def __init__(
        self,
        area_mode: AreaMode,
        demand_prediction_mode: DemandPredictionMode,
        dispatch_mode: DispatchMode,
        vehicles_number: int,
        time_periods: np.timedelta64,
        local_region_bound: LocalRegionBound,
        side_length_meter: int,
        vehicles_server_meter: int,
        neighbor_can_server: bool,
    ):

        # Component
        self.dispatch_module: DispatchModuleInterface = self.__load_dispatch_component(
            dispatch_mode=dispatch_mode
        )
        self.demand_predictor_module = None
        self.node_manager: NodeManager = None

        # Statistical variables
        self.static_service = StaticsService()

        # Data variable
        self.areas: List[Area] = None
        self.orders: List[Order] = None
        self.vehicles: List[Vehicle] = None
        self.__cost_map: np.ndarray = None
        self.node_id_to_area: Mapping[int, Area] = {}
        self.transition_temp_prool: List = []

        self.local_region_bound = local_region_bound

        # Weather data
        # TODO: MUST CHANGE
        # ------------------------------------------
        # fmt: off
        self.weather_type = np.array([2,1,1,1,1,0,1,2,1,1,3,3,3,3,3,
                                     3,3,0,0,0,2,1,1,1,1,0,1,0,1,1,
                                     1,3,1,1,0,2,2,1,0,0,2,3,2,2,2,
                                     1,2,2,2,1,0,0,2,2,2,1,2,1,1,1])
        self.minimum_temperature = np.array([12,12,11,12,14,12,9,8,7,8,9,7,9,10,11,
                                            12,13,13,11,11,11,6,5,5,4,4,6,6,5,6])
        self.maximum_temperature = np.array([17,19,19,20,20,19,13,12,13,15,16,18,18,19,19,
                                            18,20,21,19,20,19,12,9,9,10,13,12,12,13,15])
        self.wind_direction = np.array([1,2,0,2,7,6,3,2,3,7,1,0,7,1,7,
                                       0,0,7,0,7,7,7,0,7,5,7,6,6,7,7])
        self.wind_power = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                   1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])
        # fmt: on
        self.weather_type = self.__normalization_1d(self.weather_type)
        self.minimum_temperature = self.__normalization_1d(self.minimum_temperature)
        self.maximum_temperature = self.__normalization_1d(self.maximum_temperature)
        self.wind_direction = self.__normalization_1d(self.wind_direction)
        self.wind_power = self.__normalization_1d(self.wind_power)
        # ------------------------------------------

        # Input parameters
        self.area_mode: AreaMode = area_mode
        self.dispatch_mode: DispatchMode = dispatch_mode
        self.vehicles_number: int = vehicles_number
        self.time_periods: np.timedelta64 = time_periods
        self.side_length_meter: int = side_length_meter
        self.vehicle_service_meter: int = vehicles_server_meter
        self.num_grid_width: int = None
        self.num_grid_height: int = None
        self.neighbor_server_deep_limit = None

        # Control variable
        self.neighbor_can_server = neighbor_can_server

        # Process variable
        self.real_exp_time = None
        self.now_order = None
        self.step = None
        self.episode = 0

        self.__calculate_the_scale_of_devision()

        # Demand predictor variable
        self.demand_prediction_mode: DemandPredictionMode = demand_prediction_mode
        self.supply_expect = None

    @property
    def map_west_bound(self):
        return self.local_region_bound.west_bound

    @property
    def map_east_bound(self):
        return self.local_region_bound.east_bound

    @property
    def map_south_bound(self):
        return self.local_region_bound.south_bound

    @property
    def map_north_bound(self):
        return self.local_region_bound.north_bound

    @property
    def num_areas(self) -> Optional[int]:
        if (self.num_grid_width is None) or (self.num_grid_height is None):
            return None
        else:
            return self.num_grid_width * self.num_grid_height

    @property
    def total_width(self) -> float:
        return self.map_east_bound - self.map_west_bound

    @property
    def total_height(self) -> float:
        return self.map_north_bound - self.map_south_bound

    @property
    def interval_width(self) -> float:
        return self.total_width / self.num_grid_width

    @property
    def interval_height(self) -> float:
        return self.total_height / self.num_grid_height

    def create_all_instantiate(self, order_file_date: str = "0601") -> None:
        print("Read all files")
        (
            node_df,
            self.node_id_list,
            orders_df,
            vehicles,
            self.__cost_map,
        ) = read_all_files(order_file_date)
        self.node_manager = NodeManager(node_df)

        if self.area_mode == AreaMode.CLUSTER:
            print("Create Clusters")
            self.areas = self.__create_cluster()
        elif self.area_mode == AreaMode.GRID:
            print("Create Grids")
            self.areas = self.__create_grid()

        for node_id in tqdm(self.node_manager.node_id_list):
            for area in self.areas:
                for node in area.nodes:
                    if node_id == node.id:
                        self.node_id_to_area[node_id] = area

        print("Create Orders set")
        self.orders: List[Order] = [
            Order(
                id=order_row["ID"],
                release_time=order_row["Start_time"],
                pick_up_node_id=order_row["NodeS"],
                delivery_node_id=order_row["NodeE"],
                pick_up_time_window=order_row["Start_time"] + PICKUPTIMEWINDOW,
                pick_up_wait_time=None,
                arrive_info=None,
                order_value=None,
            )
            for _, order_row in orders_df.iterrows()
        ]

        # Calculate the value of all orders in advance
        # -------------------------------
        print("Pre-calculated order value")
        for each_order in self.orders:
            each_order.order_value = self.__road_cost(
                start_node_index=self.node_manager.get_node_index(
                    each_order.pick_up_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    each_order.delivery_node_id
                ),
            )
        # -------------------------------

        # Select number of vehicles
        # -------------------------------
        vehicles = vehicles[: self.vehicles_number]
        # -------------------------------

        print("Create Vehicles set")
        self.vehicles: List[Vehicle] = [
            Vehicle(
                id=vehicle_row[0],
                location_node_id=None,
                area=None,
                orders=[],
                delivery_node_id=None,
            )
            for vehicle_row in vehicles
        ]
        self.__init_vehicles_into_area()

    def reload(self, order_file_date="0601"):
        """
        Read a new order into the simulator and
        reset some variables of the simulator
        """
        print(
            "Load order " + order_file_date + "and reset the experimental environment"
        )

        self.static_service.reset()

        self.orders = None
        self.transition_temp_prool.clear()

        self.real_exp_time = None
        self.now_order = None
        self.step = None

        # read orders
        # -----------------------------------------
        order_df = read_order(
            input_file_path=base_data_path
            / TRAIN
            / f"order_2016{str(order_file_date)}.csv"
        )
        self.orders = [
            Order(
                id=order_row["ID"],
                release_time=order_row["Start_time"],
                pick_up_node_id=order_row["NodeS"],
                delivery_node_id=order_row["NodeE"],
                pick_up_time_window=order_row[1] + PICKUPTIMEWINDOW,
                pick_up_wait_time=None,
                arrive_info=None,
                order_value=None,
            )
            for _, order_row in order_df.iterrows()
        ]

        # Rename orders'ID
        # -------------------------------
        for i in range(len(self.orders)):
            self.orders[i].id = i
        # -------------------------------

        # Calculate the value of all orders in advance
        # -------------------------------
        for each_order in self.orders:
            each_order.order_value = self.__road_cost(
                start_node_index=self.node_manager.get_node_index(
                    each_order.pick_up_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    each_order.delivery_node_id
                ),
            )
        # -------------------------------

        # Reset the areas and Vehicles
        # -------------------------------
        for area in self.areas:
            area.reset()

        for vehicle in self.vehicles:
            vehicle.reset()

        self.__init_vehicles_into_area()
        # -------------------------------

        return

    def reset(self):
        print("Reset the experimental environment")

        self.static_service.reset()

        self.transition_temp_prool.clear()
        self.real_exp_time = None
        self.now_order = None
        self.step = None

        # Reset the Orders and Clusters and Vehicles
        # -------------------------------
        for order in self.orders:
            order.reset()

        for area in self.areas:
            area.reset()

        for vehicle in self.vehicles:
            vehicle.reset()

        self.__init_vehicles_into_area()
        # -------------------------------
        return

    def __init_vehicles_into_area(self) -> None:
        print("Initialization Vehicles into Clusters or Grids")
        for vehicle in self.vehicles:
            random_node = random.choice(self.node_manager.get_nodes())
            vehicle.location_node_id = random_node.id
            vehicle.area = self.node_id_to_area[random_node.id]
            vehicle.area.idle_vehicles.append(vehicle)

    def __load_dispatch_component(self, dispatch_mode: DispatchMode) -> DispatchModuleInterface:
        if dispatch_mode == DispatchMode.RANDOM:
            dispatch_module = RandomDispatch()
        return dispatch_module

    def __road_cost(self, start_node_index: int, end_node_index: int) -> int:
        return int(self.__cost_map[start_node_index][end_node_index])

    def __calculate_the_scale_of_devision(self) -> None:

        average_longitude = (self.map_east_bound - self.map_west_bound) / 2
        average_latitude = (self.map_north_bound - self.map_south_bound) / 2

        self.num_grid_width = int(
            haversine(
                self.map_west_bound,
                average_latitude,
                self.map_east_bound,
                average_latitude,
            )
            / self.side_length_meter
            + 1
        )
        self.num_grid_height = int(
            haversine(
                average_longitude,
                self.map_south_bound,
                average_longitude,
                self.map_north_bound,
            )
            / self.side_length_meter
            + 1
        )

        self.neighbor_server_deep_limit = int(
            (self.vehicle_service_meter - (0.5 * self.side_length_meter))
            // self.side_length_meter
        )

        print("----------------------------")
        print("The width of each grid", self.side_length_meter, "meters")
        print("Vehicle service range", self.vehicle_service_meter, "meters")
        print("Number of grids in east-west direction", self.num_grid_width)
        print("Number of grids in north-south direction", self.num_grid_height)
        print("Number of grids", self.num_areas)
        print("----------------------------")

    def __is_order_in_limit_region(self, order: Order) -> bool:
        if not order.pick_up_node_id in self.node_id_list:
            return False
        if not order.delivery_node_id in self.node_id_list:
            return False

        return True

    def __create_grid(self) -> List[Area]:
        node_id_list = self.node_manager.node_id_list
        node_dict = {}
        for i in tqdm(range(len(node_id_list))):
            node_dict[
                (
                    self.node_manager.node_locations[i][0],
                    self.node_manager.node_locations[i][1],
                )
            ] = self.node_id_list.index(node_id_list[i])

        all_grid: List[Area] = [
            Grid(
                id=i,
                nodes=[],
                neighbor=[],
                rebalance_number=0,
                idle_vehicles=[],
                vehicles_arrive_time={},
                orders=[],
            )
            for i in range(self.num_areas)
        ]

        for node in self.node_manager.get_nodes():
            relatively_longitude = node.longitude - self.map_west_bound
            now_grid_width_num = int(relatively_longitude // self.interval_width)
            assert now_grid_width_num <= self.num_grid_width - 1

            relatively_latitude = node.latitude - self.map_south_bound
            now_grid_height_num = int(relatively_latitude // self.interval_height)
            assert now_grid_height_num <= self.num_grid_height - 1

            all_grid[
                self.num_grid_width * now_grid_height_num + now_grid_width_num
            ].nodes.append(node)
        # ------------------------------------------------------

        # Add neighbors to each grid
        # ------------------------------------------------------
        for grid in all_grid:

            # Bound Check
            # ----------------------------
            up_neighbor = True
            down_neighbor = True
            left_neighbor = True
            right_neighbor = True
            left_up_neighbor = True
            left_down_neighbor = True
            right_up_neighbor = True
            right_down_neighbor = True

            if grid.id >= self.num_grid_width * (self.num_grid_height - 1):
                up_neighbor = False
                left_up_neighbor = False
                right_up_neighbor = False
            if grid.id < self.num_grid_width:
                down_neighbor = False
                left_down_neighbor = False
                right_down_neighbor = False
            if grid.id % self.num_grid_width == 0:
                left_neighbor = False
                left_up_neighbor = False
                left_down_neighbor = False
            if (grid.id + 1) % self.num_grid_width == 0:
                right_neighbor = False
                right_up_neighbor = False
                right_down_neighbor = False
            # ----------------------------

            # Add all neighbors
            # ----------------------------
            if up_neighbor:
                grid.neighbor.append(all_grid[grid.id + self.num_grid_width])
            if down_neighbor:
                grid.neighbor.append(all_grid[grid.id - self.num_grid_width])
            if left_neighbor:
                grid.neighbor.append(all_grid[grid.id - 1])
            if right_neighbor:
                grid.neighbor.append(all_grid[grid.id + 1])
            if left_up_neighbor:
                grid.neighbor.append(all_grid[grid.id + self.num_grid_width - 1])
            if left_down_neighbor:
                grid.neighbor.append(all_grid[grid.id - self.num_grid_width - 1])
            if right_up_neighbor:
                grid.neighbor.append(all_grid[grid.id + self.num_grid_width + 1])
            if right_down_neighbor:
                grid.neighbor.append(all_grid[grid.id - self.num_grid_width + 1])
            # ----------------------------

        # You can draw every grid(red) and neighbor(random color) here
        # ----------------------------------------------
        """
        for i in range(len(AllGrid)):
            print("Grid ID ",i,AllGrid[i])
            print(AllGrid[i].neighbor)
            self.draw_one_cluster(Cluster = AllGrid[i],random = False,show = False)
            
            for j in AllGrid[i].neighbor:
                if j.id == AllGrid[i].id :
                    continue
                print(j.id)
                self.draw_one_cluster(Cluster = j,random = True,show = False)
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()
        """
        # ----------------------------------------------
        return all_grid

    def __create_cluster(self) -> List[Area]:
        node_location: np.ndarray = self.node[["Longitude", "Latitude"]].values.round(7)
        node_id_list: np.ndarray = self.node["NodeID"].values.astype("int64")

        N = {}
        for i in range(len(node_id_list)):
            N[(node_location[i][0], node_location[i][1])] = node_id_list[i]

        clusters = [
            Cluster(
                id=i,
                nodes=[],
                neighbor=[],
                rebalance_number=0,
                idle_vehicles=[],
                vehicles_arrive_time={},
                orders=[],
            )
            for i in range(self.num_areas)
        ]

        cluster_path = (
            "./data/"
            + str(self.local_region_bound)
            + str(self.num_areas)
            + str(self.area_mode)
            + "Clusters.csv"
        )
        if os.path.exists(cluster_path):
            reader = pd.read_csv(cluster_path, chunksize=1000)
            label_pred: List = []
            for chunk in reader:
                label_pred.append(chunk)
            label_pred_df: pd.DataFrame = pd.concat(label_pred)
            label_pred: np.ndarray = label_pred_df.values
            label_pred = label_pred.flatten()
            label_pred = label_pred.astype("int64")
        else:
            raise Exception("Cluster Path not found")

        # Loading Clustering results into simulator
        print("Loading Clustering results")
        for i in range(self.num_areas):
            temp = node_location[label_pred == i]
            for j in range(len(temp)):
                clusters[i].nodes.append(
                    (
                        self.node_id_list.index(N[(temp[j, 0], temp[j, 1])]),
                        (temp[j, 0], temp[j, 1]),
                    )
                )

        save_cluster_neighbor_path = (
            "./data/"
            + str(self.local_region_bound)
            + str(self.num_areas)
            + str(self.area_mode)
            + "Neighbor.csv"
        )

        if not os.path.exists(save_cluster_neighbor_path):
            print("Computing Neighbor relationships between clusters")

            all_neighbor_list: List[Area] = []
            for cluster_1 in clusters:
                neighbor_list: List[Area] = []
                for cluster_2 in clusters:
                    if cluster_1 == cluster_2:
                        continue
                    else:
                        tmp_sum_cost = 0
                        for node_1 in cluster_1.nodes:
                            for node_2 in cluster_2.nodes:
                                tmp_sum_cost += self.__road_cost(
                                    start_node_index=self.node_manager.get_node_index(
                                        node_1.id
                                    ),
                                    end_node_index=self.node_manager.get_node_index(
                                        node_2.id
                                    ),
                                )
                        if (len(cluster_1.nodes) * len(cluster_2.nodes)) == 0:
                            road_network_distance = 99999
                        else:
                            road_network_distance = tmp_sum_cost / (
                                len(cluster_1.nodes) * len(cluster_2.nodes)
                            )

                    neighbor_list.append((cluster_2, road_network_distance))

                neighbor_list.sort(key=lambda X: X[1])

                all_neighbor_list.append([])
                for neighbor in neighbor_list:
                    all_neighbor_list[-1].append((neighbor[0].id, neighbor[1]))

            all_neighbor_df = pd.DataFrame(all_neighbor_list)
            all_neighbor_df.to_csv(
                save_cluster_neighbor_path, header=0, index=0
            )  # 不保存列名
            print(
                "Save the Neighbor relationship records to: "
                + save_cluster_neighbor_path
            )

        print("Load Neighbor relationship records")
        reader = pd.read_csv(save_cluster_neighbor_path, header=None, chunksize=1000)
        neighbor_list = []
        for chunk in reader:
            neighbor_list.append(chunk)
        neighbor_list_df: pd.DataFrame = pd.concat(neighbor_list)
        neighbor_list = neighbor_list_df.values

        id_to_cluster = {}
        for cluster in clusters:
            id_to_cluster[cluster.id] = cluster

        connected_threshold = 15
        for i in range(len(clusters)):
            for j in neighbor_list[i]:
                temp = eval(j)
                if len(clusters[i].neighbor) < 4:
                    clusters[i].neighbor.append(id_to_cluster[temp[0]])
                elif temp[1] < connected_threshold:
                    clusters[i].neighbor.append(id_to_cluster[temp[0]])
                else:
                    continue
        del id_to_cluster

        # You can draw every cluster(red) and neighbor(random color) here
        # ----------------------------------------------
        """
        for i in range(len(Clusters)):
            print("Cluster ID ",i,Clusters[i])
            print(Clusters[i].neighbor)
            self.draw_one_cluster(Cluster = Clusters[i],random = False,show = False)
            for j in Clusters[i].neighbor:
                if j.id == Clusters[i].id :
                    continue
                print(j.id)
                self.draw_one_cluster(Cluster = j,random = True,show = False)
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()
        """
        # ----------------------------------------------

        return clusters

    def __load_demand_prediction(self):
        if self.demand_prediction_mode == DemandPredictionMode.TRAINING:
            self.demand_predictor_module = None
            return

        elif self.demand_prediction_mode == DemandPredictionMode.HA:
            self.demand_predictor_module = HAPredictionModel()
            demand_prediction_model_path = (
                "./model/"
                + str(self.demand_prediction_mode)
                + "PredictionModel"
                + str(self.area_mode)
                + str(self.side_length_meter)
                + str(self.local_region_bound)
                + ".csv"
            )
        # You can extend the predictor here
        # elif self.demand_prediction_mode == 'Your predictor name':
        else:
            raise Exception("DemandPredictionMode Name error")

        if os.path.exists(demand_prediction_model_path):
            self.demand_predictor_module.Load(demand_prediction_model_path)
        else:
            print(demand_prediction_model_path)
            raise Exception("No Demand Prediction Model")
        return

    def __normalization_1d(self, arr: np.ndarray) -> np.ndarray:
        arrmax = arr.max()
        arrmin = arr.min()
        arrmaxmin = arrmax - arrmin
        result = []
        for x in arr:
            x = float(x - arrmin) / arrmaxmin
            result.append(x)

        return np.array(result)

    # Visualization tools
    # -----------------------------------------------
    def randomcolor(self) -> str:
        color_arr = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]
        color = ""
        for i in range(6):
            color += color_arr[random.randint(0, len(color_arr) - 1)]
        return "#" + color

    def draw_all_area_internal_nodes(self) -> None:
        connection_map = (read_map("./data/Map__.csv"),)
        connection_map = connection_map[0]

        areas_color = []
        for _ in range(len(self.areas)):
            areas_color.append(self.randomcolor())

        for i in tqdm(self.node_manager.node_id_list):
            for j in range(self.node_manager.node_id_list):
                if i == j:
                    continue

                if connection_map[i][j] <= 3000:
                    LX = [
                        self.node_manager.get_node(i).longitude,
                        self.node_manager.get_node(j).longitude,
                    ]
                    LY = [
                        self.node_manager.get_node(i).latitude,
                        self.node_manager.get_node(j).latitude,
                    ]

                    if self.node_id_to_area[i] == self.node_id_to_area[j]:
                        plt.plot(
                            LX,
                            LY,
                            c=areas_color[self.node_id_to_area[i].id],
                            linewidth=0.8,
                            alpha=0.5,
                        )
                    else:
                        plt.plot(LX, LY, c="grey", linewidth=0.5, alpha=0.4)

        plt.xlim(self.map_west_bound, self.map_east_bound)
        plt.ylim(self.map_south_bound, self.map_north_bound)
        plt.title(self.area_mode)
        plt.show()

    def draw_all_nodes(self) -> None:
        connection_map = (read_map("./data/Map__.csv"),)
        connection_map = connection_map[0]

        areas_color = []
        for _ in range(len(self.areas)):
            areas_color.append(self.randomcolor())

        for i in self.node_id_list:
            for j in self.node_id_list:
                if i == j:
                    continue

                if connection_map[i][j] <= 3000:
                    LX = [
                        self.node_manager.get_node(i).longitude,
                        self.node_manager.get_node(j).longitude,
                    ]
                    LY = [
                        self.node_manager.get_node(i).latitude,
                        self.node_manager.get_node(j).latitude,
                    ]

                    plt.plot(
                        LX,
                        LY,
                        c=areas_color[self.node_id_to_area[i].id],
                        linewidth=0.8,
                        alpha=0.5,
                    )

        plt.xlim(self.map_west_bound, self.map_east_bound)
        plt.ylim(self.map_south_bound, self.map_north_bound)
        plt.title(self.area_mode)
        plt.show()

    def draw_one_area(self, area: Area, random=True, show=False) -> None:
        randomc = self.randomcolor()
        for node in area.nodes:
            if random == True:
                plt.scatter(node[1][0], node[1][1], s=3, c=randomc, alpha=0.5)
            else:
                plt.scatter(node[1][0], node[1][1], s=3, c="r", alpha=0.5)
        if show == True:
            plt.xlim(self.map_west_bound, self.map_east_bound)
            plt.ylim(self.map_south_bound, self.map_north_bound)
            plt.show()

    def draw_all_vehicles(self) -> None:
        for area in self.areas:
            for vehicle in area.idle_vehicles:
                res = self.node_manager.get_node(vehicle.location_node_id)
                X = res.longitude
                Y = res.latitude
                plt.scatter(X, Y, s=3, c="b", alpha=0.3)

            for vehicle in area.vehicles_arrive_time:
                res = self.node_manager.get_node(vehicle.location_node_id)
                X = res.longitude
                Y = res.latitude
                if len(vehicle.orders):
                    plt.scatter(X, Y, s=3, c="r", alpha=0.3)
                else:
                    plt.scatter(X, Y, s=3, c="g", alpha=0.3)

        plt.xlim(self.map_west_bound, self.map_east_bound)
        plt.xlabel("red = running  blue = idle  green = Dispatch")
        plt.ylim(self.map_south_bound, self.map_north_bound)
        plt.title("Vehicles Location")
        plt.show()

    def draw_vehicle_trajectory(self, vehicle: Vehicle) -> None:
        node_1 = self.node_manager.get_node(vehicle.location_node_id)
        X1, Y1 = node_1.longitude, node_1.latitude
        node_2 = self.node_manager.get_node(vehicle.delivery_node_id)
        X2, Y2 = node_2.longitude, node_2.latitude
        # start location
        plt.scatter(X1, Y1, s=3, c="black", alpha=0.3)
        # destination
        plt.scatter(X2, Y2, s=3, c="blue", alpha=0.3)
        # Vehicles Trajectory
        LX1 = [X1, X2]
        LY1 = [Y1, Y2]
        plt.plot(LY1, LX1, c="k", linewidth=0.3, alpha=0.5)
        plt.title("Vehicles Trajectory")
        plt.show()

    # -----------------------------------------------

    def __workday_or_weekend(self, day) -> str:
        if type(day) != type(0) or day < 0 or day > 6:
            raise Exception("input format error")
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"

    ############################################################################

    # The main modules
    # ---------------------------------------------------------------------------
    def __demand_predict_function(self) -> None:
        """
        Here you can implement your own order forecasting method
        to provide efficient and accurate help for Dispatch method
        """
        return

    def __supply_expect_function(self) -> None:
        """
        Calculate the number of idle Vehicles in the next time slot
        of each cluster due to the completion of the order
        """
        self.supply_expect = np.zeros(self.num_areas)
        for area in self.areas:
            for vehicle, time in list(area.vehicles_arrive_time.items()):
                # key = Vehicle ; value = Arrivetime
                if (
                    time <= self.real_exp_time + self.time_periods
                    and len(vehicle.orders) > 0
                ):
                    self.supply_expect[area.id] += 1

    def __dispatch_function(self) -> None:
        """
        Here you can implement your own Dispatch method to
        move idle vehicles in each cluster to other clusters
        """
        for area in self.areas:
            for vehicles in area.idle_vehicles:
                self.dispatch_module()
        return

    def __match_function(self) -> None:
        """
        Each matching module will match the orders that will occur within the current time slot.
        The matching module will find the nearest idle vehicles for each order. It can also enable
        the neighbor car search system to determine the search range according to the set search distance
        and the size of the grid. It use dfs to find the nearest idle vehicles in the area.
        """

        # Count the number of idle vehicles before matching
        for area in self.areas:
            area.per_match_idle_vehicles = len(area.idle_vehicles)

        while self.now_order.release_time < self.real_exp_time + self.time_periods:

            if self.now_order.id == self.orders[-1].id:
                break

            self.static_service.increment_order_num()
            now_area: Area = self.node_id_to_area[self.now_order.pick_up_node_id]
            now_area.orders.append(self.now_order)

            if len(now_area.idle_vehicles) or len(now_area.neighbor):
                tmp_min = None

                if len(now_area.idle_vehicles):

                    # Find a nearest car to match the current order
                    # --------------------------------------
                    for vehicle in now_area.idle_vehicles:
                        tmp_road_cost = self.__road_cost(
                            start_node_index=self.node_manager.get_node_index(
                                vehicle.location_node_id
                            ),
                            end_node_index=self.node_manager.get_node_index(
                                self.now_order.pick_up_node_id
                            ),
                        )
                        if tmp_min == None:
                            tmp_min = (vehicle, tmp_road_cost, now_area)
                        elif tmp_road_cost < tmp_min[1]:
                            tmp_min = (vehicle, tmp_road_cost, now_area)
                    # --------------------------------------
                # Neighbor car search system to increase search range
                elif self.neighbor_can_server and len(now_area.neighbor):
                    tmp_min = self.__find_server_vehicle_function(
                        neighbor_server_deep_limit=self.neighbor_server_deep_limit,
                        visit_list={},
                        area=now_area,
                        tmp_min=None,
                        deep=0,
                    )

                # When all Neighbor Cluster without any idle Vehicles
                if tmp_min == None or tmp_min[1] > PICKUPTIMEWINDOW:
                    self.static_service.increment_reject_num()
                    self.now_order.arrive_info = ArriveInfo.REJECT
                # Successfully matched a vehicle
                else:
                    now_vehicle: Vehicle = tmp_min[0]
                    self.now_order.pick_up_wait_time = tmp_min[1]
                    now_vehicle.orders.append(self.now_order)

                    self.static_service.add_totally_wait_time(
                        self.__road_cost(
                            start_node_index=self.node_manager.get_node_index(
                                now_vehicle.location_node_id
                            ),
                            end_node_index=self.node_manager.get_node_index(
                                self.now_order.pick_up_node_id
                            ),
                        )
                    )

                    schedule_cost = self.__road_cost(
                        start_node_index=self.node_manager.get_node_index(
                            now_vehicle.location_node_id
                        ),
                        end_node_index=self.node_manager.get_node_index(
                            self.now_order.pick_up_node_id
                        ),
                    ) + self.__road_cost(
                        start_node_index=self.node_manager.get_node_index(
                            self.now_order.pick_up_node_id
                        ),
                        end_node_index=self.node_manager.get_node_index(
                            self.now_order.delivery_node_id
                        ),
                    )

                    # Add a destination to the current vehicle
                    now_vehicle.delivery_node_id = self.now_order.delivery_node_id

                    # Delivery Cluster {Vehicle:ArriveTime}
                    self.areas[
                        self.node_id_to_area[self.now_order.delivery_node_id].id
                    ].vehicles_arrive_time[
                        now_vehicle
                    ] = self.real_exp_time + np.timedelta64(
                        schedule_cost * MINUTES
                    )

                    # delete now Cluster's recode about now Vehicle
                    tmp_min[2].idle_vehicles.remove(now_vehicle)

                    self.now_order.arrive_info = ArriveInfo.SUCCESS
            else:
                # None available idle Vehicles
                self.static_service.increment_reject_num()
                self.now_order.arrive_info = ArriveInfo.REJECT

            # The current order has been processed and start processing the next order
            # ------------------------------
            self.now_order = self.orders[self.now_order.id + 1]

    def __find_server_vehicle_function(
        self, neighbor_server_deep_limit, visit_list, area: Area, tmp_min, deep
    ):
        """
        Use dfs visit neighbors and find nearest idle Vehicle
        """
        if deep > neighbor_server_deep_limit or area.id in visit_list:
            return tmp_min

        visit_list[area.id] = True
        for vehicle in area.idle_vehicles:
            tmp_road_cost = self.__road_cost(
                start_node_index=self.node_manager.get_node_index(
                    vehicle.location_node_id
                ),
                end_node_index=self.node_manager.get_node_index(
                    self.now_order.pick_up_node_id
                ),
            )
            if tmp_min == None:
                tmp_min = (vehicle, tmp_road_cost, area)
            elif tmp_road_cost < tmp_min[1]:
                tmp_min = (vehicle, tmp_road_cost, area)

        if self.neighbor_can_server:
            for j in area.neighbor:
                tmp_min = self.__find_server_vehicle_function(
                    neighbor_server_deep_limit,
                    visit_list,
                    j,
                    tmp_min,
                    deep + 1,
                )
        return tmp_min

    def __reward_function(self) -> None:
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your reward function here
        """
        return

    def __update_function(self) -> None:
        """
        Each time slot update Function will update each cluster
        in the simulator, processing orders and vehicles
        """
        for area in self.areas:
            # Records array of orders cleared for the last time slot
            area.orders.clear()
            for vehicle, time in list(area.vehicles_arrive_time.items()):
                # key = Vehicle ; value = Arrivetime
                if time <= self.real_exp_time:
                    # update Order
                    if len(vehicle.orders):
                        vehicle.orders[0].arrive_order_time_record(self.real_exp_time)
                    # update Vehicle info
                    vehicle.arrive_vehicle_update(area)
                    # update Cluster record
                    area.arrive_cluster_update(vehicle)

    def __get_next_state_function(self) -> None:
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your next State function here
        """
        return

    def __learning_function(self) -> None:
        return

    def __call__(self) -> None:
        self.real_exp_time = self.orders[0].release_time - self.time_periods

        # To complete running orders
        end_time = self.orders[-1].release_time + 3 * self.time_periods

        self.now_order = self.orders[0]
        self.step = 0

        episode_start_time = datetime.datetime.now()
        print("Start experiment")
        print("----------------------------")
        while self.real_exp_time <= end_time:

            step_update_start_time = datetime.datetime.now()
            self.__update_function()
            self.static_service.add_totally_update_time(
                datetime.datetime.now() - step_update_start_time
            )

            step_match_start_time = datetime.datetime.now()
            self.__match_function()
            self.static_service.add_totally_match_time(
                datetime.datetime.now() - step_match_start_time
            )

            step_reward_start_time = datetime.datetime.now()
            self.__reward_function()
            self.static_service.add_totally_reward_time(
                datetime.datetime.now() - step_reward_start_time
            )

            step_next_state_start_time = datetime.datetime.now()
            self.__get_next_state_function()
            self.static_service.add_totally_next_state_time(
                datetime.datetime.now() - step_next_state_start_time
            )
            for area in self.areas:
                area.dispatch_number = 0

            step_learning_start_time = datetime.datetime.now()
            self.__learning_function()
            self.static_service.add_totally_learning_time(
                datetime.datetime.now() - step_learning_start_time
            )

            step_demand_predict_start_time = datetime.datetime.now()
            self.__demand_predict_function()
            self.__supply_expect_function()
            self.static_service.add_totally_demand_predict_time(
                datetime.datetime.now() - step_demand_predict_start_time
            )

            # Count the number of idle vehicles before Dispatch
            for area in self.areas:
                area.per_dispatch_idle_vehicles = len(area.idle_vehicles)
            step_dispatch_start_time = datetime.datetime.now()
            self.__dispatch_function()
            self.static_service.add_totally_dispatch_time(
                datetime.datetime.now() - step_dispatch_start_time
            )
            # Count the number of idle vehicles after Dispatch
            for area in self.areas:
                area.later_dispatch_idle_vehicles = len(area.idle_vehicles)

            # A time slot is processed
            self.step += 1
            self.real_exp_time += self.time_periods
        # ------------------------------------------------
        episode_end_time = datetime.datetime.now()

        sum_order_value = 0
        order_value_num = 0
        for order in self.orders:
            if order.arrive_info != ArriveInfo.REJECT:
                sum_order_value += order.order_value
                order_value_num += 1

        # ------------------------------------------------
        print("Experiment over")
        print(f"Episode: {self.episode}")
        print(f"Clusting mode: {self.area_mode}")
        print(f"Demand Prediction mode: {self.demand_prediction_mode}")
        print(f"Dispatch mode: {self.dispatch_mode}")
        print(
            "Date: "
            + str(self.orders[0].release_time.month)
            + "/"
            + str(self.orders[0].release_time.day)
        )
        print(
            "Weekend or Workday: "
            + self.__workday_or_weekend(self.orders[0].release_time.weekday())
        )
        if self.area_mode == AreaMode.CLUSTER:
            print("Number of Clusters: " + str(self.num_areas))
        elif self.area_mode == AreaMode.GRID:
            print("Number of Grids: " + str(self.num_areas))
        print("Number of Vehicles: " + str(len(self.vehicles)))
        print("Number of Orders: " + str(len(self.orders)))
        print("Number of Reject: " + str(self.static_service.reject_num))
        print("Number of Dispatch: " + str(self.static_service.dispatch_num))
        if (self.static_service.dispatch_num) != 0:
            print(
                "Average Dispatch Cost: "
                + str(
                    self.static_service.totally_dispatch_cost
                    / self.static_service.dispatch_num
                )
            )
        if (len(self.orders) - self.static_service.reject_num) != 0:
            print(
                "Average wait time: "
                + str(
                    self.static_service.totally_wait_time
                    / (len(self.orders) - self.static_service.reject_num)
                )
            )
        print("Totally Order value: " + str(sum_order_value))
        print("Totally Update Time : " + str(self.static_service.totally_update_time))
        print(
            "Totally NextState Time : "
            + str(self.static_service.totally_next_state_time)
        )
        print(
            "Totally Learning Time : " + str(self.static_service.totally_learning_time)
        )
        print(
            "Totally Demand Predict Time : "
            + str(self.static_service.totally_demand_predict_time)
        )
        print(
            "Totally Dispatch Time : " + str(self.static_service.totally_dispatch_time)
        )
        print(
            "Totally Simulation Time : " + str(self.static_service.totally_match_time)
        )
        print("Episode Run time : " + str(episode_end_time - episode_start_time))


"""
Experiment over
Episode: 0
Clusting mode: AreaMode.GRID
Demand Prediction mode: DemandPredictionMode.TRAINING
Dispatch mode: Simulation
Date: 6/1
Weekend or Workday: Workday
Number of Grids: 1
Number of Vehicles: 6000
Number of Orders: 130
Number of Reject: 0
Number of Dispatch: 0
Average wait time: 0.0
Totally Order value: 263
Totally Update Time : 0:00:00.000383
Totally NextState Time : 0:00:00.000068
Totally Learning Time : 0:00:00.000053
Totally Demand Predict Time : 0:00:00.000438
Totally Dispatch Time : 0:00:00.000064
Totally Simulation Time : 0:00:02.311889
Episode Run time : 0:00:02.313628

(Pdb) self.areas[0].nodes[:5]
[(0, (-74.0151098, 40.706165)), (1, (-74.015079, 40.7062557)), (2, (-74.0150681, 40.7062526)), (3, (-74.0154267, 40.7082794)), (4, (-74.0155846, 40.7078839))]


"""

"""
Experiment over
Episode: 0
Clusting mode: AreaMode.GRID
Demand Prediction mode: DemandPredictionMode.TRAINING
Dispatch mode: Simulation
Date: 6/1
Weekend or Workday: Workday
Number of Grids: 1
Number of Vehicles: 6000
Number of Orders: 130
Number of Reject: 0
Number of Dispatch: 0
Average wait time: 0.0
Totally Order value: 263
Totally Update Time : 0:00:00.000609
Totally NextState Time : 0:00:00.000067
Totally Learning Time : 0:00:00.000069
Totally Demand Predict Time : 0:00:00.000643
Totally Dispatch Time : 0:00:00.000058
Totally Simulation Time : 0:00:02.335432
Episode Run time : 0:00:02.337787
"""
