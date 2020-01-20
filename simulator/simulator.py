import os
import sys
import random
import re
import copy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import datetime as dt
from math import radians, cos, sin, asin, sqrt
from datetime import datetime,timedelta
from objects.objects import Cluster,Order,Vehicle,Transition,Grid
from config.setting import *
from preprocessing.readfiles import *
###########################################################################

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

    def __init__(self,ClusterMode,DemandPredictionMode,
                DispatchMode,VehiclesNumber,TimePeriods,LocalRegionBound,
                SideLengthMeter,VehiclesServiceMeter,
                NeighborCanServer,FocusOnLocalRegion):

        #Component
        self.DispatchModule = None
        self.DemandPredictorModule = None

        #Statistical variables
        self.OrderNum = 0
        self.RejectNum = 0
        self.DispatchNum = 0
        self.TotallyDispatchCost = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyRewardTime = dt.timedelta()
        self.TotallyNextStateTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyDispatchTime = dt.timedelta()
        self.TotallyMatchTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()

        #Data variable
        self.Clusters = None
        self.Orders = None
        self.Vehicles = None
        self.Map = None
        self.Node = None
        self.Path = None
        self.NodeIDList = None
        self.NodeID2Cluseter = {}
        self.NodeID2NodesLocation = {}
        self.TransitionTempPool = []

        self.MapWestBound = LocalRegionBound[0]
        self.MapEastBound = LocalRegionBound[1]
        self.MapSouthBound = LocalRegionBound[2]
        self.MapNorthBound = LocalRegionBound[3]

        #Weather data
        #------------------------------------------
        self.WeatherType = np.array([2,1,1,1,1,0,1,2,1,1,3,3,3,3,3,
                                     3,3,0,0,0,2,1,1,1,1,0,1,0,1,1,
                                     1,3,1,1,0,2,2,1,0,0,2,3,2,2,2,
                                     1,2,2,2,1,0,0,2,2,2,1,2,1,1,1])
        self.MinimumTemperature = np.array([12,12,11,12,14,12,9,8,7,8,9,7,9,10,11,
                                            12,13,13,11,11,11,6,5,5,4,4,6,6,5,6])
        self.MaximumTemperature = np.array([17,19,19,20,20,19,13,12,13,15,16,18,18,19,19,
                                            18,20,21,19,20,19,12,9,9,10,13,12,12,13,15])
        self.WindDirection = np.array([1,2,0,2,7,6,3,2,3,7,1,0,7,1,7,
                                       0,0,7,0,7,7,7,0,7,5,7,6,6,7,7])
        self.WindPower = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                   1,1,1,1,1,1,2,1,1,1,1,1,1,1,1])
        self.WeatherType = self.Normaliztion_1D(self.WeatherType)
        self.MinimumTemperature = self.Normaliztion_1D(self.MinimumTemperature)
        self.MaximumTemperature = self.Normaliztion_1D(self.MaximumTemperature)
        self.WindDirection = self.Normaliztion_1D(self.WindDirection)
        self.WindPower = self.Normaliztion_1D(self.WindPower)
        #------------------------------------------

        #Input parameters
        self.ClusterMode = ClusterMode
        self.DispatchMode = DispatchMode
        self.VehiclesNumber = VehiclesNumber
        self.TimePeriods = TimePeriods 
        self.LocalRegionBound = LocalRegionBound
        self.SideLengthMeter = SideLengthMeter
        self.VehiclesServiceMeter = VehiclesServiceMeter
        self.ClustersNumber = None
        self.NumGrideWidth = None
        self.NumGrideHeight = None
        self.NeighborServerDeepLimit = None

        #Control variable
        self.NeighborCanServer = NeighborCanServer
        self.FocusOnLocalRegion = FocusOnLocalRegion

        #Process variable
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None
        self.Episode = 0

        self.CalculateTheScaleOfDivision()

        #Demand predictor variable
        self.DemandPredictionMode = DemandPredictionMode
        self.SupplyExpect = None

        return


    def Reload(self,OrderFileDate="1101"):
        """
        Read a new order into the simulator and 
        reset some variables of the simulator
        """
        print("Load order " + OrderFileDate + "and reset the experimental environment")

        self.OrderNum = 0
        self.RejectNum = 0
        self.DispatchNum = 0
        self.TotallyDispatchCost = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyNextStateTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyDispatchTime = dt.timedelta()
        self.TotallyMatchTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()

        self.Orders = None
        self.TransitionTempPool.clear()

        self.RealExpTime = None
        self.NowOrder = None
        self.step = None

        #read orders
        #-----------------------------------------
        if self.FocusOnLocalRegion == False:
            Orders = ReadOrder(input_file_path="./data/test/order_2016"+ str(OrderFileDate) + ".csv")
            self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]
        else:
            SaveLocalRegionBoundOrdersPath = "./data/test/order_2016" + str(self.LocalRegionBound) + str(OrderFileDate) + ".csv"
            if os.path.exists(SaveLocalRegionBoundOrdersPath):
                Orders = ReadResetOrder(input_file_path=SaveLocalRegionBoundOrdersPath)
                self.Orders = [Order(i[0],string_pdTimestamp(i[1]),self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),string_pdTimestamp(i[1])+PICKUPTIMEWINDOW,None,None,None) for i in Orders]
            else:
                Orders = ReadOrder(input_file_path="./data/test/order_2016"+ str(OrderFileDate) + ".csv")
                self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]
                #Limit order generation area
                #-------------------------------
                for i in self.Orders[:]:
                    if self.IsOrderInLimitRegion(i) == False:
                        self.Orders.remove(i)
                #-------------------------------
                LegalOrdersSet = []
                for i in self.Orders:
                    LegalOrdersSet.append(i.ID)

                OutBoundOrdersSet = []
                for i in range(len(Orders)):
                    if not i in LegalOrdersSet:
                        OutBoundOrdersSet.append(i)

                Orders = pd.DataFrame(Orders)
                Orders = Orders.drop(OutBoundOrdersSet)
                Orders.to_csv(SaveLocalRegionBoundOrdersPath,index=0)
        #-----------------------------------------

        #Rename orders'ID
        #-------------------------------
        for i in range(len(self.Orders)):
            self.Orders[i].ID = i
        #-------------------------------

        #Calculate the value of all orders in advance
        #-------------------------------
        for EachOrder in self.Orders:
            EachOrder.OrderValue = self.RoadCost(EachOrder.PickupPoint,EachOrder.DeliveryPoint)
        #-------------------------------

        #Reset the Clusters and Vehicles
        #-------------------------------
        for i in self.Clusters:
            i.Reset()

        for i in self.Vehicles:
            i.Reset()

        self.InitVehiclesIntoCluster()
        #-------------------------------

        return

    def Reset(self):
        print("Reset the experimental environment")

        self.OrderNum = 0
        self.RejectNum = 0
        self.DispatchNum = 0
        self.TotallyDispatchCost = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyNextStateTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyDispatchTime = dt.timedelta()
        self.TotallyMatchTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()
        
        self.TransitionTempPool.clear()
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None

        #Reset the Orders and Clusters and Vehicles
        #-------------------------------
        for i in self.Orders:
            i.Reset()

        for i in self.Clusters:
            i.Reset()

        for i in self.Vehicles:
            i.Reset()

        self.InitVehiclesIntoCluster()
        #-------------------------------
        return

    def InitVehiclesIntoCluster(self):
        print("Initialization Vehicles into Clusters or Grids")
        for i in self.Vehicles:
            while True:
                RandomNode = random.choice(range(len(self.Node)))
                if RandomNode in self.NodeID2Cluseter:
                    i.LocationNode = RandomNode
                    i.Cluster = self.NodeID2Cluseter[i.LocationNode]
                    i.Cluster.IdleVehicles.append(i)
                    break

    def LoadDispatchComponent(self,DispatchModule):
        self.DispatchModule = DispatchModule

    def RoadCost(self,start,end):
        return int(self.Map[start][end])

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        #haversine
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371
        return c * r * 1000

    def CalculateTheScaleOfDivision(self):
        EastWestSpan = self.LocalRegionBound[1] - self.LocalRegionBound[0]
        NorthSouthSpan = self.LocalRegionBound[3] - self.LocalRegionBound[2]

        AverageLongitude = (self.MapEastBound-self.MapWestBound)/2
        AverageLatitude = (self.MapNorthBound-self.MapSouthBound)/2

        self.NumGrideWidth = int(self.haversine(self.MapWestBound,AverageLatitude,self.MapEastBound,AverageLatitude) / self.SideLengthMeter + 1)
        self.NumGrideHeight = int(self.haversine(AverageLongitude,self.MapSouthBound,AverageLongitude,self.MapNorthBound) / self.SideLengthMeter + 1)

        self.NeighborServerDeepLimit = int((self.VehiclesServiceMeter - (0.5 * self.SideLengthMeter))//self.SideLengthMeter)
        self.ClustersNumber = self.NumGrideWidth * self.NumGrideHeight

        print("----------------------------")
        print("Map extent",self.LocalRegionBound)
        print("The width of each grid",self.SideLengthMeter,"meters")
        print("Vehicle service range",self.VehiclesServiceMeter,"meters")
        print("Number of grids in east-west direction",self.NumGrideWidth)
        print("Number of grids in north-south direction",self.NumGrideHeight)
        print("Number of grids",self.ClustersNumber)
        print("----------------------------")
        return

    def CreateAllInstantiate(self,OrderFileDate="1101"):
        print("Read all files")
        self.Node,self.NodeIDList,Orders,Vehicles,self.Map = ReadAllFiles(OrderFileDate)

        if self.ClusterMode != "Grid":
            print("Create Clusters")
            self.Clusters = self.CreateCluster()
        elif self.ClusterMode == "Grid":
            print("Create Grids")
            self.Clusters = self.CreateGrid()

        #Construct NodeID to Cluseter map for Fast calculation
        NodeID = self.Node['NodeID'].values
        for i in range(len(NodeID)):
            NodeID[i] = self.NodeIDList.index(NodeID[i])
        for i in NodeID:
            for j in self.Clusters:
                for k in j.Nodes:
                    if i == k[0]:
                        self.NodeID2Cluseter[i] = j

        print("Create Orders set")
        self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]

        #Limit order generation area
        #-------------------------------
        if self.FocusOnLocalRegion == True:
            print("Remove out-of-bounds Orders")
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        #-------------------------------

        #Calculate the value of all orders in advance
        #-------------------------------
        print("Pre-calculated order value")
        for EachOrder in self.Orders:
            EachOrder.OrderValue = self.RoadCost(EachOrder.PickupPoint,EachOrder.DeliveryPoint)
        #-------------------------------

        #Select number of vehicles
        #-------------------------------
        Vehicles = Vehicles[:self.VehiclesNumber]
        #-------------------------------

        print("Create Vehicles set")
        self.Vehicles = [Vehicle(i[0],self.NodeIDList.index(i[1]),None,[],None) for i in Vehicles]
        self.InitVehiclesIntoCluster()

        return

    def IsOrderInLimitRegion(self,Order):
        if not Order.PickupPoint in self.NodeID2NodesLocation:
            return False
        if not Order.DeliveryPoint in self.NodeID2NodesLocation:
            return False

        return True

    def IsNodeInLimitRegion(self,TempNodeList):
        if TempNodeList[0][0] < self.LocalRegionBound[0] or TempNodeList[0][0] > self.LocalRegionBound[1]:
            return False
        elif TempNodeList[0][1] < self.LocalRegionBound[2] or TempNodeList[0][1] > self.LocalRegionBound[3]:
            return False

        return True


    def CreateGrid(self):
        NumGrideHeight = self.NumGrideHeight
        NumGride = self.NumGrideWidth * self.NumGrideHeight

        NodeLocation = self.Node[['Longitude','Latitude']].values.round(7)
        NodeID = self.Node['NodeID'].values.astype('int64')

        #Select small area simulation
        #----------------------------------------------------
        if self.FocusOnLocalRegion == True:
            NodeLocation = NodeLocation.tolist()
            NodeID = NodeID.tolist()

            TempNodeList = []
            for i in range(len(NodeLocation)):
                TempNodeList.append((NodeLocation[i],NodeID[i]))

            for i in TempNodeList[:]:
                if self.IsNodeInLimitRegion(i) == False:
                    TempNodeList.remove(i)

            NodeLocation.clear()
            NodeID.clear()

            for i in TempNodeList:
                NodeLocation.append(i[0])
                NodeID.append(i[1])

            NodeLocation = np.array(NodeLocation)
        #--------------------------------------------------

        NodeSet = {}
        for i in range(len(NodeID)):
            NodeSet[(NodeLocation[i][0],NodeLocation[i][1])] = self.NodeIDList.index(NodeID[i])

        #Build each grid
        #------------------------------------------------------
        if self.FocusOnLocalRegion == True:
            TotalWidth = self.LocalRegionBound[1] - self.LocalRegionBound[0]
            TotalHeight = self.LocalRegionBound[3] - self.LocalRegionBound[2]
        else:
            TotalWidth = self.MapEastBound - self.MapWestBound 
            TotalHeight = self.MapNorthBound - self.MapSouthBound

        IntervalWidth = TotalWidth / self.NumGrideWidth
        IntervalHeight = TotalHeight / self.NumGrideHeight

        AllGrid = [Grid(i,[],[],0,[],{},[]) for i in range(NumGride)]

        for key,value in NodeSet.items():
            NowGridWidthNum = None
            NowGridHeightNum = None

            for i in range(self.NumGrideWidth):
                if self.FocusOnLocalRegion == True:
                    LeftBound = (self.LocalRegionBound[0] + i * IntervalWidth)
                    RightBound = (self.LocalRegionBound[0] + (i+1) * IntervalWidth)
                else:
                    LeftBound = (self.MapWestBound + i * IntervalWidth)
                    RightBound = (self.MapWestBound + (i+1) * IntervalWidth)

                if key[0] > LeftBound and key[0] < RightBound:
                    NowGridWidthNum = i
                    break

            for i in range(self.NumGrideHeight):
                if self.FocusOnLocalRegion == True:
                    DownBound = (self.LocalRegionBound[2] + i * IntervalHeight)
                    UpBound = (self.LocalRegionBound[2] + (i+1) * IntervalHeight)
                else:
                    DownBound = (self.MapSouthBound + i * IntervalHeight)
                    UpBound = (self.MapSouthBound + (i+1) * IntervalHeight)

                if key[1] > DownBound and key[1] < UpBound:
                    NowGridHeightNum = i
                    break

            if NowGridWidthNum == None or NowGridHeightNum == None :
                print(key[0],key[1])
                raise Exception('error')
            else:
                AllGrid[self.NumGrideWidth * NowGridHeightNum + NowGridWidthNum].Nodes.append((value,(key[0],key[1])))
        #------------------------------------------------------

        for i in AllGrid:
            for j in i.Nodes:
                self.NodeID2NodesLocation[j[0]] = j[1]

        #Add neighbors to each grid
        #------------------------------------------------------
        for i in AllGrid:

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

            if i.ID >= self.NumGrideWidth * (self.NumGrideHeight - 1):
                UpNeighbor = False
                LeftUpNeighbor = False
                RightUpNeighbor = False
            if i.ID < self.NumGrideWidth:
                DownNeighbor = False
                LeftDownNeighbor = False
                RightDownNeighbor = False
            if i.ID % self.NumGrideWidth == 0:
                LeftNeighbor = False
                LeftUpNeighbor = False
                LeftDownNeighbor = False
            if (i.ID+1) % self.NumGrideWidth == 0:
                RightNeighbor = False
                RightUpNeighbor = False
                RightDownNeighbor = False
            #----------------------------

            #Add all neighbors
            #----------------------------
            if UpNeighbor:
                i.Neighbor.append(AllGrid[i.ID+self.NumGrideWidth])
            if DownNeighbor:
                i.Neighbor.append(AllGrid[i.ID-self.NumGrideWidth])
            if LeftNeighbor:
                i.Neighbor.append(AllGrid[i.ID-1])
            if RightNeighbor:
                i.Neighbor.append(AllGrid[i.ID+1])
            if LeftUpNeighbor:
                i.Neighbor.append(AllGrid[i.ID+self.NumGrideWidth-1])
            if LeftDownNeighbor:
                i.Neighbor.append(AllGrid[i.ID-self.NumGrideWidth-1])
            if RightUpNeighbor:
                i.Neighbor.append(AllGrid[i.ID+self.NumGrideWidth+1])
            if RightDownNeighbor:
                i.Neighbor.append(AllGrid[i.ID-self.NumGrideWidth+1])
            #----------------------------

        #You can draw every grid(red) and neighbor(random color) here
        #----------------------------------------------
        '''
        for i in range(len(AllGrid)):
            print("Grid ID ",i,AllGrid[i])
            print(AllGrid[i].Neighbor)
            self.PrintCluster(Cluster = AllGrid[i],random = False,show = False)
            
            for j in AllGrid[i].Neighbor:
                if j.ID == AllGrid[i].ID :
                    continue
                print(j.ID)
                self.PrintCluster(Cluster = j,random = True,show = False)
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()
        '''
        #----------------------------------------------
        return AllGrid


    def CreateCluster(self):

        NodeLocation = self.Node[['Longitude','Latitude']].values.round(7)
        NodeID = self.Node['NodeID'].values.astype('int64')

        #Set Nodes In Limit Region
        #----------------------------------------
        if self.FocusOnLocalRegion == True:
            print("Remove out-of-bounds Nodes")
            NodeLocation = NodeLocation.tolist()
            NodeID = NodeID.tolist()

            TempNodeList = []
            for i in range(len(NodeLocation)):
                TempNodeList.append((NodeLocation[i],NodeID[i]))

            for i in TempNodeList[:]:
                if self.IsNodeInLimitRegion(i) == False:
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

        Clusters=[Cluster(i,[],[],0,[],{},[]) for i in range(self.ClustersNumber)]

        ClusterPath = './data/'+str(self.LocalRegionBound)+str(self.ClustersNumber)+str(self.ClusterMode)+'Clusters.csv'
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
        for i in range(self.ClustersNumber):
            temp = NodeLocation[label_pred == i]
            for j in range(len(temp)):
                Clusters[i].Nodes.append((self.NodeIDList.index(N[(temp[j,0],temp[j,1])]),(temp[j,0],temp[j,1])))

        SaveClusterNeighborPath = './data/'+str(self.LocalRegionBound)+str(self.ClustersNumber)+str(self.ClusterMode)+'Neighbor.csv'

        if not os.path.exists(SaveClusterNeighborPath):
            print("Computing Neighbor relationships between clusters")

            AllNeighborList = []
            for i in Clusters:
                NeighborList = []
                for j in Clusters:
                    if i == j:
                        continue
                    else:
                        TempSumCost = 0
                        for k in i.Nodes:
                            for l in j.Nodes:
                                TempSumCost += self.RoadCost(k[0],l[0])
                        if (len(i.Nodes)*len(j.Nodes)) == 0:
                            RoadNetworkDistance = 99999
                        else:
                            RoadNetworkDistance = TempSumCost / (len(i.Nodes)*len(j.Nodes))

                    NeighborList.append((j,RoadNetworkDistance))
                
                NeighborList.sort(key=lambda X: X[1])

                AllNeighborList.append([])
                for j in NeighborList:
                    AllNeighborList[-1].append((j[0].ID,j[1]))

            AllNeighborList = pd.DataFrame(AllNeighborList)
            AllNeighborList.to_csv(SaveClusterNeighborPath,header=0,index=0) #不保存列名
            print("Save the Neighbor relationship records to: "+SaveClusterNeighborPath)

        print("Load Neighbor relationship records")
        reader = pd.read_csv(SaveClusterNeighborPath,header = None,chunksize = 1000)
        NeighborList = []
        for chunk in reader:
            NeighborList.append(chunk)
        NeighborList = pd.concat(NeighborList)
        NeighborList = NeighborList.values

        ID2Cluseter = {}
        for i in Clusters:
            ID2Cluseter[i.ID] = i

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

        #self.NodeID2NodesLocation = {}
        print("Store node coordinates for drawing")
        for i in Clusters:
            for j in i.Nodes:
                self.NodeID2NodesLocation[j[0]] = j[1]

        #You can draw every cluster(red) and neighbor(random color) here
        #----------------------------------------------
        '''
        for i in range(len(Clusters)):
            print("Cluster ID ",i,Clusters[i])
            print(Clusters[i].Neighbor)
            self.PrintCluster(Cluster = Clusters[i],random = False,show = False)
            for j in Clusters[i].Neighbor:
                if j.ID == Clusters[i].ID :
                    continue
                print(j.ID)
                self.PrintCluster(Cluster = j,random = True,show = False)
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()
        '''
        #----------------------------------------------

        return Clusters


    def LoadDemandPrediction(self):
        if self.DemandPredictionMode == 'None' or self.DemandPredictionMode == "Training":
            self.DemandPredictorModule = None
            return

        elif self.DemandPredictionMode == 'HA':
            self.DemandPredictorModule = HAPredictionModel()
            DemandPredictionModelPath = "./model/"+str(self.DemandPredictionMode)+"PredictionModel"+str(self.ClusterMode)+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".csv"
        #You can extend the predictor here
        #elif self.DemandPredictionMode == 'Your predictor name':
        else:
            raise Exception('DemandPredictionMode Name error')

        if os.path.exists(DemandPredictionModelPath):
            self.DemandPredictorModule.Load(DemandPredictionModelPath)
        else:
            print(DemandPredictionModelPath)
            raise Exception("No Demand Prediction Model")
        return


    def Normaliztion_1D(self,arr):
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
    def randomcolor(self):
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0,len(colorArr)-1)]
        return "#"+color

    def DrawAllClusterInternalNodes(self):
        ConnectionMap = ReadMap('./data/Map__.csv'),
        ConnectionMap = ConnectionMap[0]

        ClusetersColor = []
        for i in range(len(self.Clusters)):
            ClusetersColor.append(self.randomcolor())

        NodeNumber = len(self.Node)
        for i in tqdm(range(NodeNumber)):
            if not i in self.NodeID2NodesLocation:
                continue
            for j in range(NodeNumber):
                if not j in self.NodeID2NodesLocation:
                    continue
                if i == j:
                    continue

                if ConnectionMap[i][j] <= 3000:
                    LX = [self.NodeID2NodesLocation[i][0],self.NodeID2NodesLocation[j][0]]
                    LY = [self.NodeID2NodesLocation[i][1],self.NodeID2NodesLocation[j][1]]

                    if self.NodeID2Cluseter[i] == self.NodeID2Cluseter[j]:
                        plt.plot(LX,LY,c=ClusetersColor[self.NodeID2Cluseter[i].ID],linewidth=0.8,alpha = 0.5)
                    else:
                        plt.plot(LX,LY,c='grey',linewidth=0.5,alpha = 0.4)

        plt.xlim(self.MapWestBound , self.MapEastBound)
        plt.ylim(self.MapSouthBound , self.MapNorthBound)
        plt.title(self.ClusterMode)
        plt.show()
        return

    def DrawAllNodes(self):
        ConnectionMap = ReadMap('./data/Map__.csv'),
        ConnectionMap = ConnectionMap[0]

        ClusetersColor = []
        for i in range(len(self.Clusters)):
            ClusetersColor.append(self.randomcolor())

        NodeNumber = len(self.Node)
        for i in range(NodeNumber):
            if not i in self.NodeID2NodesLocation:
                continue
            for j in range(NodeNumber):
                if not j in self.NodeID2NodesLocation:
                    continue
                if i == j:
                    continue

                if ConnectionMap[i][j] <= 3000:
                    LX = [self.NodeID2NodesLocation[i][0],self.NodeID2NodesLocation[j][0]]
                    LY = [self.NodeID2NodesLocation[i][1],self.NodeID2NodesLocation[j][1]]

                    plt.plot(LX,LY,c=ClusetersColor[self.NodeID2Cluseter[i].ID],linewidth=0.8,alpha = 0.5)

        plt.xlim(self.MapWestBound , self.MapEastBound)
        plt.ylim(self.MapSouthBound , self.MapNorthBound)
        plt.title(self.ClusterMode)
        plt.show()
        return

    def DrawOneCluster(self,Cluster,random=True):
        randomc = self.randomcolor()
        for i in Cluster.Nodes:
            if random == True:
                plt.scatter(i[1][0],i[1][1],s = 3, c=randomc,alpha = 0.5)
            else :
                plt.scatter(i[1][0],i[1][1],s = 3, c='r',alpha = 0.5)
        plt.xlim(self.MapWestBound , self.MapEastBound)
        plt.ylim(self.MapSouthBound , self.MapNorthBound)
        plt.title(str(Cluster.ID) + " Cluster")
        plt.show()
        return

    def DrawAllVehicles(self):
        for i in self.Clusters:
            for j in i.IdleVehicles:
                res = self.NodeID2NodesLocation[j.LocationNode]
                X = res[0]
                Y = res[1]
                plt.scatter(X,Y,s = 3, c='b',alpha = 0.3)

            for key in i.VehiclesArrivetime:
                res = self.NodeID2NodesLocation[key.LocationNode]
                X = res[0]
                Y = res[1]
                if len(key.Orders):
                    plt.scatter(X,Y,s = 3, c='r',alpha = 0.3)
                else :
                    plt.scatter(X,Y,s = 3, c='g',alpha = 0.3)

        plt.xlim(self.MapWestBound , self.MapEastBound)
        plt.xlabel("red = running  blue = idle  green = Dispatch")
        plt.ylim(self.MapSouthBound , self.MapNorthBound)
        plt.title("Vehicles Location")
        plt.show()
        return

    def DrawVehicleTrajectory(self,Vehicle):
        X1,Y1 = self.NodeID2NodesLocation[Vehicle.LocationNode]
        X2,Y2 = self.NodeID2NodesLocation[Vehicle.DeliveryPoint]
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
        return
    #-----------------------------------------------


    def WorkdayOrWeekend(self,day):
        if type(day) != type(0) or day<0 or day > 6:
            raise Exception('input format error')
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"

    def GetTimeAndWeather(self,Order):
        Month = Order.ReleasTime.month
        Day = Order.ReleasTime.day
        Week = Order.ReleasTime.weekday()
        if Week == 5 or Week == 6:
            Weekend = 1
        else:
            Weekend = 0 
        Hour = Order.ReleasTime.hour
        Minute = Order.ReleasTime.minute

        if Month == 11:
            if Hour < 12:
                WeatherType = self.WeatherType[2*(Day-1)]
            else:
                WeatherType = self.WeatherType[2*(Day-1)+1]
        else:
            raise Exception('Month format error')

        MinimumTemperature = self.MinimumTemperature[Day-1]
        MaximumTemperature = self.MaximumTemperature[Day-1]
        WindDirection = self.WindDirection[Day-1]
        WindPower = self.WindPower[Day-1]

        return [Day,Week,Weekend,Hour,Minute,WeatherType,MinimumTemperature,MaximumTemperature,WindDirection,WindPower]

    ############################################################################


    #The main modules
    #---------------------------------------------------------------------------
    def DemandPredictFunction(self):
        """
        Here you can implement your own order forecasting method
        to provide efficient and accurate help for Dispatch method
        """
        return

    def SupplyExpectFunction(self):
        """
        Calculate the number of idle Vehicles in the next time slot
        of each cluster due to the completion of the order
        """
        self.SupplyExpect = np.zeros(self.ClustersNumber)
        for i in self.Clusters:
            for key,value in list(i.VehiclesArrivetime.items()):
                #key = Vehicle ; value = Arrivetime
                if value <= self.RealExpTime + self.TimePeriods and len(key.Orders)>0:
                    self.SupplyExpect[i.ID] += 1
        return 

    def DispatchFunction(self):
        """
        Here you can implement your own Dispatch method to 
        move idle vehicles in each cluster to other clusters
        """
        return

    def MatchFunction(self):
        """
        Each matching module will match the orders that will occur within the current time slot. 
        The matching module will find the nearest idle vehicles for each order. It can also enable 
        the neighbor car search system to determine the search range according to the set search distance 
        and the size of the grid. It use dfs to find the nearest idle vehicles in the area.
        """

        #Count the number of idle vehicles before matching
        for i in self.Clusters:
            i.PerMatchIdleVehicles = len(i.IdleVehicles)

        while self.NowOrder.ReleasTime < self.RealExpTime+self.TimePeriods :

            if self.NowOrder.ID == self.Orders[-1].ID:
                break

            self.OrderNum += 1
            NowCluster = self.NodeID2Cluseter[self.NowOrder.PickupPoint]
            NowCluster.Orders.append(self.NowOrder)

            if len(NowCluster.IdleVehicles) or len(NowCluster.Neighbor):
                TempMin = None

                if len(NowCluster.IdleVehicles):

                    #Find a nearest car to match the current order
                    #--------------------------------------
                    for i in NowCluster.IdleVehicles:
                        TempRoadCost = self.RoadCost(i.LocationNode,self.NowOrder.PickupPoint)
                        if TempMin == None :
                            TempMin = (i,TempRoadCost,NowCluster)
                        elif TempRoadCost < TempMin[1] :
                            TempMin = (i,TempRoadCost,NowCluster)
                    #--------------------------------------
                #Neighbor car search system to increase search range
                elif self.NeighborCanServer and len(NowCluster.Neighbor):
                    TempMin = self.FindServerVehicleFunction(
                                                            NeighborServerDeepLimit=self.NeighborServerDeepLimit,
                                                            Visitlist={},Cluster=NowCluster,TempMin=None,deep=0
                                                            )

                #When all Neighbor Cluster without any idle Vehicles
                if TempMin == None or TempMin[1] > PICKUPTIMEWINDOW:
                    self.RejectNum+=1
                    self.NowOrder.ArriveInfo="Reject"
                #Successfully matched a vehicle
                else:
                    NowVehicle = TempMin[0]
                    self.NowOrder.PickupWaitTime = TempMin[1]
                    NowVehicle.Orders.append(self.NowOrder)

                    self.TotallyWaitTime += self.RoadCost(NowVehicle.LocationNode,self.NowOrder.PickupPoint)

                    ScheduleCost = self.RoadCost(NowVehicle.LocationNode,self.NowOrder.PickupPoint) + self.RoadCost(self.NowOrder.PickupPoint,self.NowOrder.DeliveryPoint)

                    #Add a destination to the current vehicle
                    NowVehicle.DeliveryPoint = self.NowOrder.DeliveryPoint

                    #Delivery Cluster {Vehicle:ArriveTime}
                    self.Clusters[self.NodeID2Cluseter[self.NowOrder.DeliveryPoint].ID].VehiclesArrivetime[NowVehicle] = self.RealExpTime + np.timedelta64(ScheduleCost*MINUTES)

                    #delete now Cluster's recode about now Vehicle
                    TempMin[2].IdleVehicles.remove(NowVehicle)

                    self.NowOrder.ArriveInfo="Success"
            else:
                #None available idle Vehicles
                self.RejectNum += 1    
                self.NowOrder.ArriveInfo = "Reject"

            #The current order has been processed and start processing the next order
            #------------------------------
            self.NowOrder = self.Orders[self.NowOrder.ID+1]

        return


    def FindServerVehicleFunction(self,NeighborServerDeepLimit,Visitlist,Cluster,TempMin,deep):
        """
        Use dfs visit neighbors and find nearest idle Vehicle
        """
        if deep > NeighborServerDeepLimit or Cluster.ID in Visitlist:
            return TempMin

        Visitlist[Cluster.ID] = True
        for i in Cluster.IdleVehicles:
            TempRoadCost = self.RoadCost(i.LocationNode,self.NowOrder.PickupPoint)
            if TempMin == None :
                TempMin = (i,TempRoadCost,Cluster)
            elif TempRoadCost < TempMin[1]:
                TempMin = (i,TempRoadCost,Cluster)

        if self.NeighborCanServer:
            for j in Cluster.Neighbor:
                TempMin = self.FindServerVehicleFunction(NeighborServerDeepLimit,Visitlist,j,TempMin,deep+1)
        return TempMin


    def RewardFunction(self):
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your reward function here
        """
        return

    def UpdateFunction(self):
        """
        Each time slot update Function will update each cluster
        in the simulator, processing orders and vehicles
        """
        for i in self.Clusters:
            #Records array of orders cleared for the last time slot
            i.Orders.clear()
            for key,value in list(i.VehiclesArrivetime.items()):
                #key = Vehicle ; value = Arrivetime
                if value <= self.RealExpTime :
                    #update Order
                    if len(key.Orders):
                        key.Orders[0].ArriveOrderTimeRecord(self.RealExpTime)
                    #update Vehicle info
                    key.ArriveVehicleUpDate(i)
                    #update Cluster record
                    i.ArriveClusterUpDate(key)
        return

    def GetNextStateFunction(self):
        """
        When apply Dispatch with Reinforcement learning
        you need to implement your next State function here
        """
        return

    def LearningFunction(self):
        return

    def SimCity(self):
        self.RealExpTime = self.Orders[0].ReleasTime - self.TimePeriods

        #To complete running orders
        EndTime = self.Orders[-1].ReleasTime + 3 * self.TimePeriods

        self.NowOrder = self.Orders[0]
        self.step = 0

        EpisodeStartTime = dt.datetime.now()
        print("Start experiment")
        print("----------------------------")
        while self.RealExpTime <= EndTime:

            StepStartTime = dt.datetime.now()

            StepUpdateStartTime = dt.datetime.now()
            self.UpdateFunction()
            self.TotallyUpdateTime += dt.datetime.now() - StepUpdateStartTime

            StepMatchStartTime = dt.datetime.now()
            self.MatchFunction()
            self.TotallyMatchTime += dt.datetime.now() - StepMatchStartTime

            StepRewardStartTime = dt.datetime.now()
            self.RewardFunction()
            self.TotallyRewardTime += dt.datetime.now() - StepRewardStartTime

            StepNextStateStartTime = dt.datetime.now()
            self.GetNextStateFunction()
            self.TotallyNextStateTime += dt.datetime.now() - StepNextStateStartTime
            for i in self.Clusters:
                i.DispatchNumber = 0

            StepLearningStartTime = dt.datetime.now()
            self.LearningFunction()
            self.TotallyLearningTime += dt.datetime.now() - StepLearningStartTime

            StepDemandPredictStartTime = dt.datetime.now()
            self.DemandPredictFunction()
            self.SupplyExpectFunction()
            self.TotallyDemandPredictTime += dt.datetime.now() - StepDemandPredictStartTime  

            #Count the number of idle vehicles before Dispatch
            for i in self.Clusters:
                i.PerDispatchIdleVehicles = len(i.IdleVehicles)
            StepDispatchStartTime = dt.datetime.now()
            self.DispatchFunction()
            self.TotallyDispatchTime += dt.datetime.now() - StepDispatchStartTime  
            #Count the number of idle vehicles after Dispatch
            for i in self.Clusters:
                i.LaterDispatchIdleVehicles = len(i.IdleVehicles)

            #A time slot is processed
            self.step += 1
            self.RealExpTime += self.TimePeriods
        #------------------------------------------------
        EpisodeEndTime = dt.datetime.now()

        SumOrderValue = 0
        OrderValueNum = 0
        for i in self.Orders:
            if i.ArriveInfo != "Reject":
                SumOrderValue += i.OrderValue
                OrderValueNum += 1

        #------------------------------------------------
        print("Experiment over")
        print("Episode: " + str(self.Episode))
        print("Clusting mode: " + self.ClusterMode)
        print("Demand Prediction mode: " + self.DemandPredictionMode)
        print("Dispatch mode: " + self.DispatchMode)
        print("Date: " + str(self.Orders[0].ReleasTime.month) + "/" + str(self.Orders[0].ReleasTime.day))
        print("Weekend or Workday: " + self.WorkdayOrWeekend(self.Orders[0].ReleasTime.weekday()))
        if self.ClusterMode != "Grid":
            print("Number of Clusters: " + str(len(self.Clusters)))
        elif self.ClusterMode == "Grid":
            print("Number of Grids: " + str((self.NumGrideWidth * self.NumGrideHeight)))
        print("Number of Vehicles: " + str(len(self.Vehicles)))
        print("Number of Orders: " + str(len(self.Orders)))
        print("Number of Reject: " + str(self.RejectNum))
        print("Number of Dispatch: " + str(self.DispatchNum))
        if (self.DispatchNum)!=0:
            print("Average Dispatch Cost: " + str(self.TotallyDispatchCost/self.DispatchNum))
        if (len(self.Orders)-self.RejectNum)!=0:
            print("Average wait time: " + str(self.TotallyWaitTime/(len(self.Orders)-self.RejectNum)))
        print("Totally Order value: " + str(SumOrderValue))
        print("Totally Update Time : " + str(self.TotallyUpdateTime))
        print("Totally NextState Time : " + str(self.TotallyNextStateTime))
        print("Totally Learning Time : " + str(self.TotallyLearningTime))
        print("Totally Demand Predict Time : " + str(self.TotallyDemandPredictTime))
        print("Totally Dispatch Time : " + str(self.TotallyDispatchTime))
        print("Totally Simulation Time : " + str(self.TotallyMatchTime))
        print("Episode Run time : " + str(EpisodeEndTime - EpisodeStartTime))
        return


if __name__ == '__main__':
    DispatchMode = "Simulation"
    DemandPredictionMode = "None"
    ClusterMode = "Grid"
    EXPSIM = Simulation(
                        ClusterMode = ClusterMode,
                        DemandPredictionMode = DemandPredictionMode,
                        DispatchMode = DispatchMode,
                        VehiclesNumber = VehiclesNumber,
                        TimePeriods = TIMESTEP,
                        LocalRegionBound = LocalRegionBound,
                        SideLengthMeter = SideLengthMeter,
                        VehiclesServiceMeter = VehiclesServiceMeter,
                        NeighborCanServer = NeighborCanServer,
                        FocusOnLocalRegion = FocusOnLocalRegion,
                        )
    EXPSIM.CreateAllInstantiate()
    EXPSIM.SimCity()
