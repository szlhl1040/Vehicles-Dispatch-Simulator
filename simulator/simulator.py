import os
from typing import List, Mapping, Optional
import sys
from pathlib import Path
import random
import re
import copy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import datetime as dt
from datetime import datetime,timedelta
from objects.objects import Order,Vehicle, Transition, Area, Cluster, Grid
from config.setting import (
    MINUTES,
    TIMESTEP,
    PICKUPTIMEWINDOW,
    NeighborCanServer,
    FocusOnLocalRegion,
    LocalRegionBound,
    VehiclesNumber,
    SideLengthMeter,
    VehiclesServiceMeter,
    DispatchMode,
    DemandPredictionMode,
    AreaMode,
)
from preprocessing.readfiles import *
from util import haversine
from tqdm import tqdm
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
        area_mode,
        demand_prediction_mode,
        dispatch_mode,
        vehicles_number,
        time_periods,
        local_region_bound,
        side_length_meter,
        vehicles_server_meter,
        neighbor_can_server,
        focus_on_local_region
    ):

        #Component
        self.dispatch_module = None
        self.demand_predictor_module = None

        #Statistical variables
        self.order_num = 0
        self.reject_num = 0
        self.dispatch_num = 0
        self.totally_dispatch_cost = 0
        self.totally_wait_time = 0
        self.totally_update_time = dt.timedelta()
        self.totally_reward_time = dt.timedelta()
        self.totally_next_state_time = dt.timedelta()
        self.totally_learning_time = dt.timedelta()
        self.totally_dispatch_time = dt.timedelta()
        self.totally_match_time = dt.timedelta()
        self.totally_demand_predict_time = dt.timedelta()

        #Data variable
        self.areas: Optional[List[Area]] = None
        self.orders: Optional[List[Order]] = None
        self.vehicles: List[Vehicle] = None
        self.Map = None
        self.Node = None
        self.node_id_list = None
        self.node_id_to_area: Mapping = {}
        self.node_id_to_nodes_location: Mapping = {}
        self.transition_temp_prool: List = []

        self.map_west_bound = local_region_bound[0]
        self.map_east_bound = local_region_bound[1]
        self.map_south_bound = local_region_bound[2]
        self.map_north_bound = local_region_bound[3]

        #Weather data
        # TODO: MUST CHANGE
        #------------------------------------------
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
        self.weather_type = self.normalization_1d(self.weather_type)
        self.minimum_temperature = self.normalization_1d(self.minimum_temperature)
        self.maximum_temperature = self.normalization_1d(self.maximum_temperature)
        self.wind_direction = self.normalization_1d(self.wind_direction)
        self.wind_power = self.normalization_1d(self.wind_power)
        #------------------------------------------

        #Input parameters
        self.area_mode = area_mode
        self.dispatch_mode = dispatch_mode
        self.vehicles_number = vehicles_number
        self.time_periods = time_periods
        self.local_region_bound = local_region_bound
        self.side_length_meter = side_length_meter
        self.vehicle_service_meter = vehicles_server_meter
        self.area_number = None
        self.num_grid_width = None
        self.num_grid_height = None
        self.neighbor_server_deep_limit = None

        #Control variable
        self.neighbor_can_server = neighbor_can_server
        self.focus_on_local_region = focus_on_local_region

        #Process variable
        self.real_exp_time = None
        self.now_order = None
        self.step = None
        self.episode = 0

        self.calculate_the_scale_of_devision()

        #Demand predictor variable
        self.demand_prediction_mode = demand_prediction_mode
        self.supply_expect = None

        return


    def reload(self,order_file_date="0601"):
        """
        Read a new order into the simulator and 
        reset some variables of the simulator
        """
        print("Load order " + order_file_date + "and reset the experimental environment")

        self.order_num = 0
        self.reject_num = 0
        self.dispatch_num = 0
        self.totally_dispatch_cost = 0
        self.totally_wait_time = 0
        self.totally_update_time = dt.timedelta()
        self.totally_next_state_time = dt.timedelta()
        self.totally_learning_time = dt.timedelta()
        self.totally_dispatch_time = dt.timedelta()
        self.totally_match_time = dt.timedelta()
        self.totally_demand_predict_time = dt.timedelta()

        self.orders = None
        self.transition_temp_prool.clear()

        self.real_exp_time = None
        self.now_order = None
        self.step = None

        #read orders
        #-----------------------------------------
        # if self.focus_on_local_region == False:
        if True:
            orders = read_order(input_file_path=base_data_path / TRAIN / f"order_2016{str(order_file_date)}.csv")
            self.orders = [
                Order(i[0],i[1],self.node_id_list.index(i[2]),self.node_id_list.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in orders]
        else:
            SaveLocalRegionBoundOrdersPath = base_data_path / TRAIN / f"order_2016{str(order_file_date)}.csv"
            if os.path.exists(SaveLocalRegionBoundOrdersPath):
                orders = ReadResetOrder(input_file_path=SaveLocalRegionBoundOrdersPath)
                breakpoint()
                self.orders = [Order(i[0],string_pdTimestamp(i[1]),self.node_id_list.index(i[2]),self.node_id_list.index(i[3]),string_pdTimestamp(i[1])+PICKUPTIMEWINDOW,None,None,None) for i in orders]
            else:
                orders = read_order(input_file_path=base_data_path / TRAIN / f"order_2016{str(order_file_date)}.csv")
                self.orders = [Order(i[0],i[1],self.node_id_list.index(i[2]),self.node_id_list.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in orders]
                #Limit order generation area
                #-------------------------------
                for i in self.orders[:]:
                    if self.is_order_in_limit_region(i) == False:
                        self.orders.remove(i)
                #-------------------------------
                LegalOrdersSet = []
                for i in self.orders:
                    LegalOrdersSet.append(i.id)

                OutBoundOrdersSet = []
                for i in range(len(orders)):
                    if not i in LegalOrdersSet:
                        OutBoundOrdersSet.append(i)

                orders = pd.DataFrame(orders)
                orders = orders.drop(OutBoundOrdersSet)
                orders.to_csv(SaveLocalRegionBoundOrdersPath,index=0)
        #-----------------------------------------

        #Rename orders'ID
        #-------------------------------
        for i in range(len(self.orders)):
            self.orders[i].id = i
        #-------------------------------

        #Calculate the value of all orders in advance
        #-------------------------------
        for each_order in self.orders:
            each_order.order_value = self.road_cost(each_order.pickup_point,each_order.delivery_point)
        #-------------------------------

        #Reset the areas and Vehicles
        #-------------------------------
        for area in self.areas:
            area.reset()

        for vehicle in self.vehicles:
            vehicle.reset()

        self.init_vehicles_into_area()
        #-------------------------------

        return

    def reset(self):
        print("Reset the experimental environment")

        self.order_num = 0
        self.reject_num = 0
        self.dispatch_num = 0
        self.totally_dispatch_cost = 0
        self.totally_wait_time = 0
        self.totally_update_time = dt.timedelta()
        self.totally_next_state_time = dt.timedelta()
        self.totally_learning_time = dt.timedelta()
        self.totally_dispatch_time = dt.timedelta()
        self.totally_match_time = dt.timedelta()
        self.totally_demand_predict_time = dt.timedelta()
        
        self.transition_temp_prool.clear()
        self.real_exp_time = None
        self.now_order = None
        self.step = None

        #Reset the Orders and Clusters and Vehicles
        #-------------------------------
        for order in self.orders:
            order.reset()

        for area in self.areas:
            area.reset()

        for vehicle in self.vehicles:
            vehicle.reset()

        self.init_vehicles_into_area()
        #-------------------------------
        return

    def init_vehicles_into_area(self):
        print("Initialization Vehicles into Clusters or Grids")
        for vehicle in self.vehicles:
            while True:
                random_node = random.choice(range(len(self.Node)))
                if random_node in self.node_id_to_area:
                    vehicle.location_node = random_node
                    vehicle.area = self.node_id_to_area[vehicle.location_node]
                    vehicle.area.idle_vehicles.append(vehicle)
                    break

    def load_dispatch_component(self, dispatch_module):
        self.dispatch_module = dispatch_module

    def road_cost(self,start,end):
        return int(self.Map[start][end])

    def calculate_the_scale_of_devision(self):

        average_longitude = (self.map_east_bound-self.map_west_bound)/2
        average_latitude = (self.map_north_bound-self.map_south_bound)/2

        self.num_grid_width = int(
            haversine(
                self.map_west_bound,
                average_latitude,
                self.map_east_bound,
                average_latitude
            ) / self.side_length_meter + 1
        )
        self.num_grid_height = int(
            haversine(
                average_longitude,
                self.map_south_bound,
                average_longitude,
                self.map_north_bound
            ) / self.side_length_meter + 1
        )

        self.neighbor_server_deep_limit = int((self.vehicle_service_meter - (0.5 * self.side_length_meter))//self.side_length_meter)
        self.area_number = self.num_grid_width * self.num_grid_height

        print("----------------------------")
        print("Map extent",self.local_region_bound)
        print("The width of each grid",self.side_length_meter,"meters")
        print("Vehicle service range",self.vehicle_service_meter,"meters")
        print("Number of grids in east-west direction",self.num_grid_width)
        print("Number of grids in north-south direction",self.num_grid_height)
        print("Number of grids",self.area_number)
        print("----------------------------")
        return

    def create_all_instantiate(self, order_file_date: str = "0601"):
        print("Read all files")
        self.Node, self.node_id_list, orders_df, vehicles, self.Map = read_all_files(order_file_date)

        if self.area_mode != "Grid":
            print("Create Clusters")
            self.areas = self.create_cluster()
        elif self.area_mode == "Grid":
            print("Create Grids")
            self.areas = self.create_grid()

        #Construct NodeID to Cluseter map for Fast calculation
        node_id_list = self.Node['NodeID'].values
        for i in range(len(node_id_list)):
            node_id_list[i] = self.node_id_list.index(node_id_list[i])
        for node_id in tqdm(node_id_list):
            for area in self.areas:
                for node in area.nodes:
                    if node_id == node[0]:
                        self.node_id_to_area[node_id] = area

        print("Create Orders set")
        self.orders = [Order(order_row[0],order_row[1],self.node_id_list.index(order_row[2]),self.node_id_list.index(order_row[3]),order_row[1]+PICKUPTIMEWINDOW,None,None,None) for order_row in orders_df]

        #Limit order generation area
        #-------------------------------
        if self.focus_on_local_region == True:
            print("Remove out-of-bounds Orders")
            for order in self.orders[:]:
                if self.is_order_in_limit_region(order) == False:
                    self.orders.remove(order)
            for i in range(len(self.orders)):
                self.orders[i].id = i
        #-------------------------------

        #Calculate the value of all orders in advance
        #-------------------------------
        print("Pre-calculated order value")
        for each_order in self.orders:
            each_order.order_value = self.road_cost(each_order.pickup_point,each_order.delivery_point)
        #-------------------------------

        #Select number of vehicles
        #-------------------------------
        vehicles = vehicles[:self.vehicles_number]
        #-------------------------------

        print("Create Vehicles set")
        self.vehicles = [Vehicle(i[0],self.node_id_list.index(i[1]),None,[],None) for i in vehicles]
        self.init_vehicles_into_area()

        return

    def is_order_in_limit_region(self, order: Order):
        if not order.pickup_point in self.node_id_to_nodes_location:
            return False
        if not order.delivery_point in self.node_id_to_nodes_location:
            return False

        return True

    def is_node_in_limit_region(self,TempNodeList):
        if TempNodeList[0][0] < self.local_region_bound[0] or TempNodeList[0][0] > self.local_region_bound[1]:
            return False
        elif TempNodeList[0][1] < self.local_region_bound[2] or TempNodeList[0][1] > self.local_region_bound[3]:
            return False

        return True


    def create_grid(self):
        NumGrideHeight = self.num_grid_height
        NumGride = self.num_grid_width * self.num_grid_height

        NodeLocation = self.Node[['Longitude','Latitude']].values.round(7)
        NodeID = self.Node['NodeID'].values.astype('int64')

        #Select small area simulation
        #----------------------------------------------------
        if self.focus_on_local_region == True:
            NodeLocation = NodeLocation.tolist()
            NodeID = NodeID.tolist()

            TempNodeList = []
            for i in range(len(NodeLocation)):
                TempNodeList.append((NodeLocation[i],NodeID[i]))

            for i in TempNodeList[:]:
                if self.is_node_in_limit_region(i) == False:
                    TempNodeList.remove(i)

            NodeLocation.clear()
            NodeID.clear()

            for i in TempNodeList:
                NodeLocation.append(i[0])
                NodeID.append(i[1])

            NodeLocation = np.array(NodeLocation)
        #--------------------------------------------------
        NodeSet = {}
        for i in tqdm(range(len(NodeID))):
            NodeSet[(NodeLocation[i][0],NodeLocation[i][1])] = self.node_id_list.index(NodeID[i])

        #Build each grid
        #------------------------------------------------------
        if self.focus_on_local_region == True:
            TotalWidth = self.local_region_bound[1] - self.local_region_bound[0]
            TotalHeight = self.local_region_bound[3] - self.local_region_bound[2]
        else:
            TotalWidth = self.map_east_bound - self.map_west_bound 
            TotalHeight = self.map_north_bound - self.map_south_bound

        IntervalWidth = TotalWidth / self.num_grid_width
        IntervalHeight = TotalHeight / self.num_grid_height

        AllGrid: List[Area] = [Grid(i,[],[],0,[],{},[]) for i in range(NumGride)]

        for key,value in tqdm(NodeSet.items()):
            NowGridWidthNum = None
            NowGridHeightNum = None

            for i in range(self.num_grid_width):
                if self.focus_on_local_region == True:
                    LeftBound = (self.local_region_bound[0] + i * IntervalWidth)
                    RightBound = (self.local_region_bound[0] + (i+1) * IntervalWidth)
                else:
                    LeftBound = (self.map_west_bound + i * IntervalWidth)
                    RightBound = (self.map_west_bound + (i+1) * IntervalWidth)

                if key[0] > LeftBound and key[0] <= RightBound:
                    NowGridWidthNum = i
                    break

            for i in range(self.num_grid_height):
                if self.focus_on_local_region == True:
                    DownBound = (self.local_region_bound[2] + i * IntervalHeight)
                    UpBound = (self.local_region_bound[2] + (i+1) * IntervalHeight)
                else:
                    DownBound = (self.map_south_bound + i * IntervalHeight)
                    UpBound = (self.map_south_bound + (i+1) * IntervalHeight)

                if key[1] > DownBound and key[1] <= UpBound:
                    NowGridHeightNum = i
                    break

            if NowGridWidthNum == None or NowGridHeightNum == None :
                print(key[0],key[1])
                raise Exception('error')
            else:
                AllGrid[self.num_grid_width * NowGridHeightNum + NowGridWidthNum].nodes.append((value,(key[0],key[1])))
        #------------------------------------------------------

        for i in AllGrid:
            for j in i.nodes:
                self.node_id_to_nodes_location[j[0]] = j[1]

        #Add neighbors to each grid
        #------------------------------------------------------
        for i in tqdm(AllGrid):

            #Bound Check
            #----------------------------
            UpNeighbor = True
            DownNeighbor = True
            LeftNeighbor = True
            RightNeighbor = True
            LeftUpNeighbor = True
            LeftDownNeighbor = True
            RightUpNeighbor = True
            RightDownNeighbor = True

            if i.id >= self.num_grid_width * (self.num_grid_height - 1):
                UpNeighbor = False
                LeftUpNeighbor = False
                RightUpNeighbor = False
            if i.id < self.num_grid_width:
                DownNeighbor = False
                LeftDownNeighbor = False
                RightDownNeighbor = False
            if i.id % self.num_grid_width == 0:
                LeftNeighbor = False
                LeftUpNeighbor = False
                LeftDownNeighbor = False
            if (i.id+1) % self.num_grid_width == 0:
                RightNeighbor = False
                RightUpNeighbor = False
                RightDownNeighbor = False
            #----------------------------

            #Add all neighbors
            #----------------------------
            if UpNeighbor:
                i.Neighbor.append(AllGrid[i.id+self.num_grid_width])
            if DownNeighbor:
                i.Neighbor.append(AllGrid[i.id-self.num_grid_width])
            if LeftNeighbor:
                i.Neighbor.append(AllGrid[i.id-1])
            if RightNeighbor:
                i.Neighbor.append(AllGrid[i.id+1])
            if LeftUpNeighbor:
                i.Neighbor.append(AllGrid[i.id+self.num_grid_width-1])
            if LeftDownNeighbor:
                i.Neighbor.append(AllGrid[i.id-self.num_grid_width-1])
            if RightUpNeighbor:
                i.Neighbor.append(AllGrid[i.id+self.num_grid_width+1])
            if RightDownNeighbor:
                i.Neighbor.append(AllGrid[i.id-self.num_grid_width+1])
            #----------------------------

        #You can draw every grid(red) and neighbor(random color) here
        #----------------------------------------------
        '''
        for i in range(len(AllGrid)):
            print("Grid ID ",i,AllGrid[i])
            print(AllGrid[i].Neighbor)
            self.draw_one_cluster(Cluster = AllGrid[i],random = False,show = False)
            
            for j in AllGrid[i].Neighbor:
                if j.id == AllGrid[i].id :
                    continue
                print(j.id)
                self.draw_one_cluster(Cluster = j,random = True,show = False)
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()
        '''
        #----------------------------------------------
        return AllGrid


    def create_cluster(self):

        NodeLocation = self.Node[['Longitude','Latitude']].values.round(7)
        NodeID = self.Node['NodeID'].values.astype('int64')

        #Set Nodes In Limit Region
        #----------------------------------------
        if self.focus_on_local_region == True:
            print("Remove out-of-bounds Nodes")
            NodeLocation = NodeLocation.tolist()
            NodeID = NodeID.tolist()

            TempNodeList = []
            for i in range(len(NodeLocation)):
                TempNodeList.append((NodeLocation[i],NodeID[i]))

            for i in TempNodeList[:]:
                if self.is_node_in_limit_region(i) == False:
                    TempNodeList.remove(i)

            NodeLocation.clear()
            NodeID.clear()

            for i in TempNodeList:
                #NodeLocation.append(i[0])
                NodeLocation.append(i[0])
                NodeID.append(i[1])

            NodeLocation = np.array(NodeLocation)
        #----------------------------------------

        N = {}
        for i in range(len(NodeID)):
            N[(NodeLocation[i][0],NodeLocation[i][1])] = NodeID[i]

        Clusters=[Cluster(i,[],[],0,[],{},[]) for i in range(self.area_number)]

        ClusterPath = './data/'+str(self.local_region_bound)+str(self.area_number)+str(self.area_mode)+'Clusters.csv'
        if os.path.exists(ClusterPath):
            reader = pd.read_csv(ClusterPath,chunksize = 1000)
            label_pred = []
            for chunk in reader:
                label_pred.append(chunk)
            label_pred = pd.concat(label_pred)
            label_pred = label_pred.values
            label_pred = label_pred.flatten()
            label_pred = label_pred.astype('int64')
        else:
            raise Exception('Cluster Path not found')

        #Loading Clustering results into simulator
        print("Loading Clustering results")
        for i in range(self.area_number):
            temp = NodeLocation[label_pred == i]
            for j in range(len(temp)):
                Clusters[i].nodes.append((self.node_id_list.index(N[(temp[j,0],temp[j,1])]),(temp[j,0],temp[j,1])))

        SaveClusterNeighborPath = './data/'+str(self.local_region_bound)+str(self.area_number)+str(self.area_mode)+'Neighbor.csv'

        if not os.path.exists(SaveClusterNeighborPath):
            print("Computing Neighbor relationships between clusters")

            all_neighbor_list= []
            for cluster_1 in Clusters:
                NeighborList = []
                for cluster_2 in Clusters:
                    if cluster_1 == cluster_2:
                        continue
                    else:
                        TempSumCost = 0
                        for node_1 in cluster_1.nodes:
                            for node_2 in cluster_2.nodes:
                                TempSumCost += self.road_cost(node_1[0],node_2[0])
                        if (len(cluster_1.nodes)*len(cluster_2.nodes)) == 0:
                            RoadNetworkDistance = 99999
                        else:
                            RoadNetworkDistance = TempSumCost / (len(cluster_1.nodes)*len(cluster_2.nodes))

                    NeighborList.append((cluster_2,RoadNetworkDistance))
                
                NeighborList.sort(key=lambda X: X[1])

                all_neighbor_list.append([])
                for neighbor in NeighborList:
                    all_neighbor_list[-1].append((neighbor[0].ID,neighbor[1]))

            all_neighbor_list = pd.DataFrame(all_neighbor_list)
            all_neighbor_list.to_csv(SaveClusterNeighborPath,header=0,index=0) #不保存列名
            print("Save the Neighbor relationship records to: "+SaveClusterNeighborPath)

        print("Load Neighbor relationship records")
        reader = pd.read_csv(SaveClusterNeighborPath,header = None,chunksize = 1000)
        NeighborList = []
        for chunk in reader:
            NeighborList.append(chunk)
        NeighborList = pd.concat(NeighborList)
        NeighborList = NeighborList.values

        ID2Cluseter = {}
        for cluster in Clusters:
            ID2Cluseter[cluster.id] = cluster

        ConnectedThreshold = 15
        for i in range(len(Clusters)):
            for j in NeighborList[i]:
                temp = eval(j)
                if len(Clusters[i].Neighbor) < 4:
                    Clusters[i].Neighbor.append(ID2Cluseter[temp[0]])
                elif temp[1] < ConnectedThreshold:
                    Clusters[i].Neighbor.append(ID2Cluseter[temp[0]])
                else:
                    continue
        del ID2Cluseter

        #self.node_id_to_nodes_location = {}
        print("Store node coordinates for drawing")
        for cluster in Clusters:
            for node in cluster.nodes:
                self.node_id_to_nodes_location[node[0]] = node[1]

        #You can draw every cluster(red) and neighbor(random color) here
        #----------------------------------------------
        '''
        for i in range(len(Clusters)):
            print("Cluster ID ",i,Clusters[i])
            print(Clusters[i].Neighbor)
            self.draw_one_cluster(Cluster = Clusters[i],random = False,show = False)
            for j in Clusters[i].Neighbor:
                if j.id == Clusters[i].id :
                    continue
                print(j.id)
                self.draw_one_cluster(Cluster = j,random = True,show = False)
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()
        '''
        #----------------------------------------------

        return Clusters


    def load_demand_prediction(self):
        if self.demand_prediction_mode == 'None' or self.demand_prediction_mode == "Training":
            self.demand_predictor_module = None
            return

        elif self.demand_prediction_mode == 'HA':
            self.demand_predictor_module = HAPredictionModel()
            DemandPredictionModelPath = "./model/"+str(self.demand_prediction_mode)+"PredictionModel"+str(self.area_mode)+str(self.side_length_meter)+str(self.local_region_bound)+".csv"
        #You can extend the predictor here
        #elif self.demand_prediction_mode == 'Your predictor name':
        else:
            raise Exception('DemandPredictionMode Name error')

        if os.path.exists(DemandPredictionModelPath):
            self.demand_predictor_module.Load(DemandPredictionModelPath)
        else:
            print(DemandPredictionModelPath)
            raise Exception("No Demand Prediction Model")
        return


    def normalization_1d(self,arr: np.ndarray) -> np.ndarray:
        arrmax = arr.max()
        arrmin = arr.min()
        arrmaxmin = arrmax - arrmin
        result = []
        for x in arr:
            x = float(x - arrmin)/arrmaxmin
            result.append(x)

        return np.array(result)


    #Visualization tools
    #-----------------------------------------------
    def randomcolor(self) -> str:
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,len(colorArr)-1)]
        return "#"+color

    def draw_all_area_internal_nodes(self) -> None:
        connection_map = ReadMap('./data/Map__.csv'),
        connection_map = connection_map[0]

        areas_color = []
        for area in range(len(self.areas)):
            areas_color.append(self.randomcolor())

        NodeNumber = len(self.Node)
        for i in tqdm(range(NodeNumber)):
            if not i in self.node_id_to_nodes_location:
                continue
            for j in range(NodeNumber):
                if not j in self.node_id_to_nodes_location:
                    continue
                if i == j:
                    continue

                if connection_map[i][j] <= 3000:
                    LX = [self.node_id_to_nodes_location[i][0],self.node_id_to_nodes_location[j][0]]
                    LY = [self.node_id_to_nodes_location[i][1],self.node_id_to_nodes_location[j][1]]

                    if self.node_id_to_area[i] == self.node_id_to_area[j]:
                        plt.plot(LX,LY,c=areas_color[self.node_id_to_area[i].id],linewidth=0.8,alpha = 0.5)
                    else:
                        plt.plot(LX,LY,c='grey',linewidth=0.5,alpha = 0.4)

        plt.xlim(self.map_west_bound , self.map_east_bound)
        plt.ylim(self.map_south_bound , self.map_north_bound)
        plt.title(self.area_mode)
        plt.show()

    def draw_all_nodes(self) -> None:
        connection_map = ReadMap('./data/Map__.csv'),
        connection_map = connection_map[0]

        areas_color = []
        for i in range(len(self.Areas)):
            areas_color.append(self.randomcolor())

        node_size = len(self.Node)
        for i in range(node_size):
            if not i in self.node_id_to_nodes_location:
                continue
            for j in range(node_size):
                if not j in self.node_id_to_nodes_location:
                    continue
                if i == j:
                    continue

                if connection_map[i][j] <= 3000:
                    LX = [self.node_id_to_nodes_location[i][0],self.node_id_to_nodes_location[j][0]]
                    LY = [self.node_id_to_nodes_location[i][1],self.node_id_to_nodes_location[j][1]]

                    plt.plot(LX,LY,c=areas_color[self.node_id_to_area[i].id],linewidth=0.8,alpha = 0.5)

        plt.xlim(self.map_west_bound , self.map_east_bound)
        plt.ylim(self.map_south_bound , self.map_north_bound)
        plt.title(self.area_mode)
        plt.show()

    def draw_one_area(self,area: Area,random=True,show=False) -> None:
        randomc = self.randomcolor()
        for node in area.nodes:
            if random == True:
                plt.scatter(node[1][0],node[1][1],s = 3, c=randomc,alpha = 0.5)
            else :
                plt.scatter(node[1][0],node[1][1],s = 3, c='r',alpha = 0.5)
        if show == True:
            plt.xlim(self.map_west_bound , self.map_east_bound)
            plt.ylim(self.map_south_bound , self.map_north_bound)
            plt.show()

    def draw_all_vehicles(self) -> None:
        for area in self.areas:
            for vehicle in area.idle_vehicles:
                res = self.node_id_to_nodes_location[vehicle.location_node]
                X = res[0]
                Y = res[1]
                plt.scatter(X,Y,s = 3, c='b',alpha = 0.3)

            for key in area.vehicles_arrive_time:
                res = self.node_id_to_nodes_location[key.location_node]
                X = res[0]
                Y = res[1]
                if len(key.orders):
                    plt.scatter(X,Y,s = 3, c='r',alpha = 0.3)
                else :
                    plt.scatter(X,Y,s = 3, c='g',alpha = 0.3)

        plt.xlim(self.map_west_bound , self.map_east_bound)
        plt.xlabel("red = running  blue = idle  green = Dispatch")
        plt.ylim(self.map_south_bound , self.map_north_bound)
        plt.title("Vehicles Location")
        plt.show()

    def draw_vehicle_trajectory(self, vehicle: Vehicle) -> None:
        X1,Y1 = self.node_id_to_nodes_location[vehicle.location_node]
        X2,Y2 = self.node_id_to_nodes_location[vehicle.delivery_point]
        #start location
        plt.scatter(X1,Y1,s = 3, c='black',alpha = 0.3)
        #destination
        plt.scatter(X2,Y2,s = 3, c='blue',alpha = 0.3)
        #Vehicles Trajectory 
        LX1=[X1,X2]
        LY1=[Y1,Y2]
        plt.plot(LY1,LX1,c='k',linewidth=0.3,alpha = 0.5)
        plt.title("Vehicles Trajectory")
        plt.show()
    #-----------------------------------------------


    def workday_or_weekend(self,day) -> str:
        if type(day) != type(0) or day<0 or day > 6:
            raise Exception('input format error')
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"

    def get_time_and_weather(self,order: Order):
        month = order.ReleasTime.month
        day = order.ReleasTime.day
        week = order.ReleasTime.weekday()
        if week == 5 or week == 6:
            weekend = 1
        else:
            weekend = 0 
        hour = order.ReleasTime.hour
        minute = order.ReleasTime.minute

        if month == 11:
            if hour < 12:
                weather_type = self.weather_type[2*(day-1)]
            else:
                weather_type = self.weather_type[2*(day-1)+1]
        else:
            raise Exception('Month format error')

        minimum_temperature = self.minimum_temperature[day-1]
        maximum_temperature = self.maximum_temperature[day-1]
        wind_direction = self.wind_direction[day-1]
        wind_power = self.wind_power[day-1]

        return [day,week,weekend,hour,minute,weather_type,minimum_temperature,maximum_temperature,wind_direction,wind_power]

    ############################################################################


    #The main modules
    #---------------------------------------------------------------------------
    def demand_predict_function(self) -> None:
        """
        Here you can implement your own order forecasting method
        to provide efficient and accurate help for Dispatch method
        """
        return

    def supply_expect_function(self) -> None:
        """
        Calculate the number of idle Vehicles in the next time slot
        of each cluster due to the completion of the order
        """
        self.supply_expect = np.zeros(self.area_number)
        for area in self.areas:
            for key,value in list(area.vehicles_arrive_time.items()):
                #key = Vehicle ; value = Arrivetime
                if value <= self.real_exp_time + self.time_periods and len(key.orders)>0:
                    self.supply_expect[area.id] += 1

    def dispatch_function(self) -> None:
        """
        Here you can implement your own Dispatch method to 
        move idle vehicles in each cluster to other clusters
        """
        return

    def match_function(self) -> None:
        """
        Each matching module will match the orders that will occur within the current time slot. 
        The matching module will find the nearest idle vehicles for each order. It can also enable 
        the neighbor car search system to determine the search range according to the set search distance 
        and the size of the grid. It use dfs to find the nearest idle vehicles in the area.
        """

        #Count the number of idle vehicles before matching
        for area in self.areas:
            area.per_match_idle_vehicles = len(area.idle_vehicles)

        while self.now_order.ReleasTime < self.real_exp_time+self.time_periods :

            if self.now_order.id == self.orders[-1].id:
                break

            self.order_num += 1
            now_area: Area = self.node_id_to_area[self.now_order.pickup_point]
            now_area.orders.append(self.now_order)

            if len(now_area.idle_vehicles) or len(now_area.Neighbor):
                tmp_min = None

                if len(now_area.idle_vehicles):

                    #Find a nearest car to match the current order
                    #--------------------------------------
                    for vehicle in now_area.idle_vehicles:
                        tmp_road_cost = self.road_cost(vehicle.location_node,self.now_order.pickup_point)
                        if tmp_min == None :
                            tmp_min = (vehicle,tmp_road_cost,now_area)
                        elif tmp_road_cost < tmp_min[1] :
                            tmp_min = (vehicle,tmp_road_cost,now_area)
                    #--------------------------------------
                #Neighbor car search system to increase search range
                elif self.neighbor_can_server and len(now_area.Neighbor):
                    tmp_min = self.find_server_vehicle_function(
                                                            NeighborServerDeepLimit=self.neighbor_server_deep_limit,
                                                            Visitlist={},area=now_area,tmp_min=None,deep=0
                                                            )

                #When all Neighbor Cluster without any idle Vehicles
                if tmp_min == None or tmp_min[1] > PICKUPTIMEWINDOW:
                    self.reject_num+=1
                    self.now_order.ArriveInfo="Reject"
                #Successfully matched a vehicle
                else:
                    now_vehicle: Vehicle = tmp_min[0]
                    self.now_order.PickupWaitTime = tmp_min[1]
                    now_vehicle.orders.append(self.now_order)

                    self.totally_wait_time += self.road_cost(now_vehicle.location_node,self.now_order.pickup_point)

                    ScheduleCost = self.road_cost(now_vehicle.location_node,self.now_order.pickup_point) + self.road_cost(self.now_order.pickup_point,self.now_order.delivery_point)

                    #Add a destination to the current vehicle
                    now_vehicle.delivery_point = self.now_order.delivery_point

                    #Delivery Cluster {Vehicle:ArriveTime}
                    self.areas[self.node_id_to_area[self.now_order.delivery_point].id].vehicles_arrive_time[now_vehicle] = self.real_exp_time + np.timedelta64(ScheduleCost*MINUTES)

                    #delete now Cluster's recode about now Vehicle
                    tmp_min[2].idle_vehicles.remove(now_vehicle)

                    self.now_order.ArriveInfo="Success"
            else:
                #None available idle Vehicles
                self.reject_num += 1    
                self.now_order.ArriveInfo = "Reject"

            #The current order has been processed and start processing the next order
            #------------------------------
            self.now_order = self.orders[self.now_order.id+1]


    def find_server_vehicle_function(
        self,
        neighbor_server_deep_limit,
        visit_list,
        area: Area,
        tmp_min,
        deep
    ):
        """
        Use dfs visit neighbors and find nearest idle Vehicle
        """
        if deep > neighbor_server_deep_limit or area.id in visit_list:
            return tmp_min

        visit_list[area.id] = True
        for vehicle in area.idle_vehicles:
            tmp_road_cost = self.road_cost(vehicle.location_node,self.now_order.pickup_point)
            if tmp_min == None :
                tmp_min = (vehicle, tmp_road_cost, area)
            elif tmp_road_cost < tmp_min[1]:
                tmp_min = (vehicle, tmp_road_cost, area)

        if self.neighbor_can_server:
            for j in area.Neighbor:
                tmp_min = self.find_server_vehicle_function(
                    neighbor_server_deep_limit,
                    visit_list,
                    j,
                    tmp_min,
                    deep+1,
                )
        return tmp_min


    def reward_function(self) -> None:
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your reward function here
        """
        return

    def update_function(self) -> None:
        """
        Each time slot update Function will update each cluster
        in the simulator, processing orders and vehicles
        """
        for area in self.areas:
            #Records array of orders cleared for the last time slot
            area.orders.clear()
            for key,value in list(area.vehicles_arrive_time.items()):
                #key = Vehicle ; value = Arrivetime
                if value <= self.real_exp_time :
                    #update Order
                    if len(key.orders):
                        key.orders[0].arrive_order_time_record(self.real_exp_time)
                    #update Vehicle info
                    key.arrive_vehicle_update(area)
                    #update Cluster record
                    area.arrive_cluster_update(key)

    def get_next_state_function(self) -> None:
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your next State function here
        """
        return

    def learning_function(self) -> None:
        return

    def simulate(self) -> None:
        self.real_exp_time = self.orders[0].ReleasTime - self.time_periods

        #To complete running orders
        EndTime = self.orders[-1].ReleasTime + 3 * self.time_periods

        self.now_order = self.orders[0]
        self.step = 0

        EpisodeStartTime = dt.datetime.now()
        print("Start experiment")
        print("----------------------------")
        while self.real_exp_time <= EndTime:

            step_update_start_time= dt.datetime.now()
            self.update_function()
            self.totally_update_time += dt.datetime.now() - step_update_start_time

            step_match_start_time = dt.datetime.now()
            self.match_function()
            self.totally_match_time += dt.datetime.now() - step_match_start_time

            step_reward_start_time = dt.datetime.now()
            self.reward_function()
            self.totally_reward_time += dt.datetime.now() - step_reward_start_time

            step_next_state_start_time = dt.datetime.now()
            self.get_next_state_function()
            self.totally_next_state_time += dt.datetime.now() - step_next_state_start_time
            for area in self.areas:
                area.dispatch_number = 0

            step_learning_start_time = dt.datetime.now()
            self.learning_function()
            self.totally_learning_time += dt.datetime.now() - step_learning_start_time

            step_demand_predict_start_time= dt.datetime.now()
            self.demand_predict_function()
            self.supply_expect_function()
            self.totally_demand_predict_time += dt.datetime.now() - step_demand_predict_start_time  

            #Count the number of idle vehicles before Dispatch
            for area in self.areas:
                area.per_dispatch_idle_vehicles = len(area.idle_vehicles)
            step_dispatch_start_time = dt.datetime.now()
            self.dispatch_function()
            self.totally_dispatch_time += dt.datetime.now() - step_dispatch_start_time  
            #Count the number of idle vehicles after Dispatch
            for area in self.areas:
                area.later_dispatch_idle_vehicles = len(area.idle_vehicles)

            #A time slot is processed
            self.step += 1
            self.real_exp_time += self.time_periods
        #------------------------------------------------
        episode_end_time = dt.datetime.now()

        sum_order_value = 0
        order_value_num = 0
        for order in self.orders:
            if order.ArriveInfo != "Reject":
                sum_order_value += order.order_value
                order_value_num += 1

        #------------------------------------------------
        print("Experiment over")
        print("Episode: " + str(self.episode))
        print("Clusting mode: " + self.area_mode)
        print("Demand Prediction mode: " + self.demand_prediction_mode)
        print("Dispatch mode: " + self.dispatch_mode)
        print("Date: " + str(self.orders[0].ReleasTime.month) + "/" + str(self.orders[0].ReleasTime.day))
        print("Weekend or Workday: " + self.workday_or_weekend(self.orders[0].ReleasTime.weekday()))
        if self.area_mode != "Grid":
            print("Number of Clusters: " + str(len(self.areas)))
        elif self.area_mode == "Grid":
            print("Number of Grids: " + str((self.num_grid_width * self.num_grid_height)))
        print("Number of Vehicles: " + str(len(self.vehicles)))
        print("Number of Orders: " + str(len(self.orders)))
        print("Number of Reject: " + str(self.reject_num))
        print("Number of Dispatch: " + str(self.dispatch_num))
        if (self.dispatch_num)!=0:
            print("Average Dispatch Cost: " + str(self.totally_dispatch_cost/self.dispatch_num))
        if (len(self.orders)-self.reject_num)!=0:
            print("Average wait time: " + str(self.totally_wait_time/(len(self.orders)-self.reject_num)))
        print("Totally Order value: " + str(sum_order_value))
        print("Totally Update Time : " + str(self.totally_update_time))
        print("Totally NextState Time : " + str(self.totally_next_state_time))
        print("Totally Learning Time : " + str(self.totally_learning_time))
        print("Totally Demand Predict Time : " + str(self.totally_demand_predict_time))
        print("Totally Dispatch Time : " + str(self.totally_dispatch_time))
        print("Totally Simulation Time : " + str(self.totally_match_time))
        print("Episode Run time : " + str(episode_end_time - EpisodeStartTime))


if __name__ == '__main__':
    DispatchMode = "Simulation"
    DemandPredictionMode = "None"
    AreaMode = "Grid"
    simulator = Simulation(
        area_mode=AreaMode,
        demand_prediction_mode=DemandPredictionMode,
        dispatch_mode=DispatchMode,
        vehicles_number=VehiclesNumber,
        time_periods=TIMESTEP,
        local_region_bound=LocalRegionBound,
        side_length_meter=SideLengthMeter,
        vehicles_server_meter=VehiclesServiceMeter,
        neighbor_can_server=NeighborCanServer,
        focus_on_local_region=FocusOnLocalRegion,
    )
    simulator.create_all_instantiate()
    simulator.simulate()
