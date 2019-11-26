# -*- coding: utf-8 -*-  
import os
import sys
import random
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import numpy as np
import logging
import errno
import datetime as dt


from datetime import datetime,timedelta
#import multiprocessing as mp
from collections import deque
from objects.objects import Cluster,Order,Vehicle,Agent,Grid

from simulator.CNNDemandPrediction import CNNPredictionModel

from Cluster.TransportationCluster import TransportationCluster,TransportationCluster2,TransportationCluster3
from Cluster.OrderCluster import OrderCluster
from Cluster.SpectralCluster import SpectralCluster

from config.setting import *
from preprocessing.readfiles import *
#from tools.tools import *
#from dqn.KerasAgent import DQNAgent

from tqdm import tqdm
from matplotlib.pyplot import plot,savefig
from sklearn.cluster import KMeans
#数组转one hot用
#from keras.utils import to_categorical

#from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,AffinityPropagation

###########################################################################


class Logger(object):
    """docstring for Logger"""
    def __init__(self):
        self.logger = logging.getLogger('mylogger')
        self.logger.setLevel(logging.DEBUG)

        #fh = logging.FileHandler('test.log',mode='w',encoding='UTF-8')
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:   %(message)s')
        formatter = logging.Formatter('%(asctime)s  %(levelname)s:  %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)


    def LogTest(self):
        self.logger.debug('Hello World!')

    def LogDebug(self, message):
        self.logger.debug(message)

    def LogInfo(self, message):
        self.logger.info(message)

    def LogWarning(self, message):
        self.logger.warning(message)

    def LogError(self, message):
        self.logger.error(message)



class Simulation(object):

    def __init__(self,explog,p2pConnectedThreshold,ClusterMode,DemandPredictionMode,
                RebalanceMode,VehiclesNumber,TimePeriods,LocalRegionBound,
                SideLengthMeter,VehiclesServiceMeter,TimeAndWeatherOneHotSignal,
                NeighborCanServer,FocusOnLocalRegion,SaveMemorySignal):

        #Rebalance Component
        self.RebalanceModule = None

        #统计变量
        self.RewardValue = 50
        self.SumWorking = 0
        self.SumIdle = 0
        self.OrderNum = 0
        self.RejectNum = 0
        self.RebalanceNum = 0
        self.TotallyRebalanceCost = 0
        self.StepReward = 0
        self.TotallyMaxQ = 0
        self.TotallyMaxQNumber = 0
        self.TotallyReward = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyRememberTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyRebalanceTime = dt.timedelta()
        self.TotallySimulationTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()

        #数据变量
        self.Clusters = None
        self.Orders = None
        self.Vehicles = None
        self.Map = None
        self.Node = None
        self.Path = None
        self.NodeIDList = None
        self.SaveNumber = None
        self.NodeID2Cluseter = {}
        self.NodeID2NodesLocation = {}
        self.SaveMemory = []
        self.AgentTempPool = []
        #Weather Table
        #------------------------------------------
        #晴=0 多云=1 阴=2 小雨=3
        #一天分上下两个天气状态
        self.WeatherTable11 = [2,1,1,1,1,0,1,2,1,1,3,3,3,3,3,
                               3,3,0,0,0,2,1,1,1,1,0,1,0,1,1,
                               1,3,1,1,0,2,2,1,0,0,2,3,2,2,2,
                               1,2,2,2,1,0,0,2,2,2,1,2,1,1,1]

        self.WeatherTable12 = [2,3,2,2,1,0,0,1,0,0,0,0,0,0,0,
                               1,2,2,1,1,1,2,1,0,1,3,2,3,3,1,
                               2,2,1,0,1,1,2,2,2,3,0,0,0,1,2,
                               2,2,3,3,2,1,2,0,1,0,1,1,1,1,3,
                               2,2] 
        #------------------------------------------

        #参数变量
        self.explog = explog
        self.p2pConnectedThreshold = p2pConnectedThreshold
        self.ClusterMode = ClusterMode
        self.RebalanceMode = RebalanceMode
        self.VehiclesNumber = VehiclesNumber
        self.TimePeriods = TimePeriods 
        self.LocalRegionBound = LocalRegionBound    #(1,2,3,4) = (左,右,下,上)
        self.SideLengthMeter = SideLengthMeter
        self.VehiclesServiceMeter = VehiclesServiceMeter
        self.ClustersNumber = None
        self.NumGrideWidth = None
        self.NumGrideHeight = None
        self.NeighborServerDeepLimit = None

        #控制变量
        self.TimeAndWeatherOneHotSignal = TimeAndWeatherOneHotSignal
        self.NeighborCanServer = NeighborCanServer
        self.FocusOnLocalRegion = FocusOnLocalRegion
        self.SaveMemorySignal = SaveMemorySignal
        self.SaveClusterSignal = False

        #过程变量
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None
        self.Episode = 0

        if self.FocusOnLocalRegion == False:
            self.LocalRegionBound = (104.0110,104.1239,30.6188,30.7023)

        self.CalculateTheScaleOfDivision()

        #需求预测变量
        self.DemandPrediction = None
        self.DemandPredictionMode = DemandPredictionMode
        self.PredictionModel_learning_rate = 0.0001
        self.NumGridsDimension = 9
        self.DemandPredictionInput = None
        self.DemandPredictionOutput = None
        self.PredictionOutputDimension = self.ClustersNumber
        self.LastThreeStepOrdersMap = np.zeros(self.ClustersNumber)
        self.LastTwoStepOrdersMap = np.zeros(self.ClustersNumber)
        self.LastOneStepOrdersMap = np.zeros(self.ClustersNumber)
        self.SinDayOfWeek = np.zeros(self.ClustersNumber)
        self.CosDayOfWeek = np.zeros(self.ClustersNumber)
        self.SinHourOfDay = np.zeros(self.ClustersNumber)
        self.CosHourOfDay = np.zeros(self.ClustersNumber)
        self.SinMinuteOfHour = np.zeros(self.ClustersNumber)
        self.CosMinuteOfHour = np.zeros(self.ClustersNumber)
        self.Weather = np.zeros(self.ClustersNumber)
        self.LoadDemandPrediction()


    def Reset(self,OrderFileDate="1101"):
        self.explog.LogInfo("Reset environment")

        self.SumWorking = 0
        self.SumIdle = 0
        self.OrderNum = 0
        self.RejectNum = 0
        self.RebalanceNum = 0
        self.TotallyRebalanceCost = 0
        self.StepReward = 0
        self.TotallyMaxQ = 0
        self.TotallyMaxQNumber = 0
        self.TotallyReward = 0
        self.TotallyWaitTime = 0
        self.TotallyUpdateTime = dt.timedelta()
        self.TotallyRememberTime = dt.timedelta()
        self.TotallyLearningTime = dt.timedelta()
        self.TotallyRebalanceTime = dt.timedelta()
        self.TotallySimulationTime = dt.timedelta()
        self.TotallyDemandPredictTime = dt.timedelta()

        #数据变量
        self.Orders = None
        self.Vehicles = None
        self.SaveMemory.clear()
        self.AgentTempPool.clear()

        #过程变量
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None

        for i in self.Clusters:
            i.Reset()

        Vehicles = ReadDriver(input_file_path="./data/Drivers1101.csv")


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
                #self.Orders = [Order(i[0],string_datetime(i[1]),self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),string_datetime(i[1])+PICKUPTIMEWINDOW,None,None,None) for i in Orders]
            else:
                Orders = ReadOrder(input_file_path="./data/test/order_2016"+ str(OrderFileDate) + ".csv")
                self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]
                #限定订单发生区域
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
                Orders.to_csv(SaveLocalRegionBoundOrdersPath,index=0) #不保存列名
        #-----------------------------------------
                
        #重命名ID
        #-------------------------------
        for i in range(len(self.Orders)):
            self.Orders[i].ID = i
        #-------------------------------

        #提前计算所有订单的价值
        #-------------------------------
        for EachOrder in self.Orders:
            EachOrder.OrderValue = self.RoadCost(EachOrder.PickupPoint,EachOrder.DeliveryPoint)
        #-------------------------------

        Vehicles = Vehicles[:self.VehiclesNumber]

        self.Vehicles = [Vehicle(i[0],self.NodeIDList.index(i[1]),None,[],None) for i in Vehicles]

        #出租车位置随机化
        #-------------------------------
        NodeNum = len(self.Node)
        for i in self.Vehicles:
            i.LocationNode = random.choice(range(NodeNum))
        #-------------------------------

        #将车初始化进聚类里面
        for i in self.Vehicles:
            for j in self.Clusters:
                for k in j.Nodes:
                    if i.LocationNode == k[0]:
                        i.Cluster = j
                        j.IdleVehicles.append(i)

        return


    def LoadRebalanceComponent(self,RebalanceModule):
        self.RebalanceModule = RebalanceModule


    def CalculateTheScaleOfDivision(self):
        EastWestSpan = self.LocalRegionBound[1] - self.LocalRegionBound[0]
        NorthSouthSpan = self.LocalRegionBound[3] - self.LocalRegionBound[2]

        self.NumGrideWidth = int((EastWestSpan // (self.SideLengthMeter * Meter2Longitude)) + 1)
        self.NumGrideHeight = int((NorthSouthSpan // (self.SideLengthMeter * Meter2Latitude)) + 1)
        self.NeighborServerDeepLimit = int((self.VehiclesServiceMeter - (0.5 * self.SideLengthMeter))//self.SideLengthMeter)
        self.ClustersNumber = self.NumGrideWidth * self.NumGrideHeight

        print("----------------------------")
        print("每个格子宽度",self.SideLengthMeter,"米")
        print("车辆服务范围",self.VehiclesServiceMeter,"米")
        print("东西方向格子数为",self.NumGrideWidth)
        print("南北方向格子数为",self.NumGrideHeight)
        print("簇或格子数目",self.ClustersNumber)
        print("检索层数",self.NeighborServerDeepLimit)
        print("----------------------------")

        return

    def CalculateInterTransportation(self):
        InterTransportationRes = []

        for i in self.Clusters:
            TempRoadCost = 0
            for j in i.Nodes:
                for k in i.Nodes:
                    TempRoadCost += self.RoadCost(j[0],k[0])
            InterTransportationRes.append(TempRoadCost)

        print("InterTransportationRes :",InterTransportationRes)
        AverageInterTransportationRes = []
        SumInterTransportationRes = 0
        for i in range(len(InterTransportationRes)):
            SumInterTransportationRes += InterTransportationRes[i]
            if len(self.Clusters[i].Nodes) == 0:
                AverageInterTransportationRes.append("zero")
            else:
                AverageInterTransportationRes.append(InterTransportationRes[i]/len(self.Clusters[i].Nodes))
        print("AverageInterTransportationRes :",AverageInterTransportationRes)
        print("Sum = ",SumInterTransportationRes)
        print("AverageSum = ",SumInterTransportationRes/len(self.Clusters))
        return


    def RoadCost(self,start,end):
        return int(self.Map[start][end])


    def CreateAllInstantiate(self,OrderFileDate="1101"):
        self.explog.LogInfo("Read all files")
        self.Node,self.Path,self.NodeIDList,Orders,Vehicles,self.Map = ReadAllFiles(OrderFileDate)

        #if self.ClusterMode == "KmeansCluster" or self.ClusterMode == "TransportationCluster":
        if self.ClusterMode != "Grid":
            self.explog.LogInfo("Create Clusters")
            self.Clusters = self.CreateCluster()
            self.CreateClusterAdjacencyMatrix()
        elif self.ClusterMode == "Grid":
            self.explog.LogInfo("Create Grids")
            self.Clusters = self.CreateGrid()

        #for i in self.Clusters:
        #    print(len(i.Neighbor))


        self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]

        #限定订单发生区域
        #-------------------------------
        if self.FocusOnLocalRegion == True:
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        #-------------------------------

        #提前计算所有订单的价值
        #-------------------------------
        for EachOrder in self.Orders:
            EachOrder.OrderValue = self.RoadCost(EachOrder.PickupPoint,EachOrder.DeliveryPoint)
        #-------------------------------

        #限定车辆数目
        #-------------------------------
        Vehicles = Vehicles[:self.VehiclesNumber]
        #-------------------------------

        self.Vehicles = [Vehicle(i[0],self.NodeIDList.index(i[1]),None,[],None) for i in Vehicles]

        #出租车位置随机化
        self.explog.LogInfo("Random initialization Vehicles' Location")
        #-------------------------------
        NodeNum = len(self.Node)
        for i in self.Vehicles:
            i.LocationNode = random.choice(range(NodeNum))
        #-------------------------------


        #将车初始化进聚类里面
        self.explog.LogInfo("Initialization Vehicles into Clusters or Grids")
        for i in self.Vehicles:
            for j in self.Clusters:
                for k in j.Nodes:
                    if i.LocationNode == k[0]:
                        i.Cluster = j
                        j.IdleVehicles.append(i)

        NodeID = self.Node['NodeID'].values

        for i in range(len(NodeID)):
            NodeID[i] = self.NodeIDList.index(NodeID[i])

        #self.NodeID2Cluseter = {}
        for i in NodeID:
            for j in self.Clusters:
                for k in j.Nodes:
                    #print(i,type(i),self.NodeIDList.index(i),type(self.NodeIDList.index(i)),k,type(k),k[0],type(k[0]),j,type(j))
                    if i == k[0]:
                        self.NodeID2Cluseter[i] = j
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
        NodeID = self.Node['NodeID'].values.astype('int')

        #self.FocusOnLocalRegion = False
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
                #NodeLocation.append(i[0])
                NodeLocation.append(i[0])
                NodeID.append(i[1])

            NodeLocation = np.array(NodeLocation)

        NodeSet = {}
        for i in range(len(NodeID)):
            NodeSet[(NodeLocation[i][0],NodeLocation[i][1])] = self.NodeIDList.index(NodeID[i])

        #装载Rode进每一个Grid
        #------------------------------------------------------
        if self.FocusOnLocalRegion == True:
            TotalWidth = self.LocalRegionBound[1] - self.LocalRegionBound[0]
            TotalHeight = self.LocalRegionBound[3] - self.LocalRegionBound[2]
        else:
            TotalWidth = 104.13 - 104.00767
            TotalHeight = 30.7092 - 30.6119

        IntervalWidth = TotalWidth / self.NumGrideWidth
        IntervalHeight = TotalHeight / self.NumGrideHeight

        #每个格子长宽
        #print(IntervalWidth,IntervalHeight)

        AllGrid = [Grid(i,[],[],0,[],{},[]) for i in range(NumGride)]

        for key,value in NodeSet.items():
            #print(key[0],key[1],value)

            NowGridWidthNum = None
            NowGridHeightNum = None

            # 问题在i
            for i in range(self.NumGrideWidth):
                if self.FocusOnLocalRegion == True:
                    LeftBound = (self.LocalRegionBound[0] + i * IntervalWidth)
                    RightBound = (self.LocalRegionBound[0] + (i+1) * IntervalWidth)
                else:
                    LeftBound = (104.007 + i * IntervalWidth)
                    RightBound = (104.007 + (i+1) * IntervalWidth)

                if key[0] > LeftBound and key[0] < RightBound:
                    NowGridWidthNum = i
                    break

            for i in range(self.NumGrideHeight):
                if self.FocusOnLocalRegion == True:
                    DownBound = (self.LocalRegionBound[2] + i * IntervalHeight)
                    UpBound = (self.LocalRegionBound[2] + (i+1) * IntervalHeight)
                else:
                    DownBound = (30.6119 + i * IntervalHeight)
                    UpBound = (30.6119 + (i+1) * IntervalHeight)

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



        #给每一个Grid安排Neighbor
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
                #没上则没左右上
                LeftUpNeighbor = False
                RightUpNeighbor = False
            if i.ID < self.NumGrideWidth:
                DownNeighbor = False
                #没下则没左右下
                LeftDownNeighbor = False
                RightDownNeighbor = False
            if i.ID % self.NumGrideWidth == 0:
                LeftNeighbor = False
                #没左则没左上左下
                LeftUpNeighbor = False
                LeftDownNeighbor = False
            if (i.ID+1) % self.NumGrideWidth == 0:
                RightNeighbor = False
                #没右则没右上右下
                RightUpNeighbor = False
                RightDownNeighbor = False
            #----------------------------

            #check
            #----------------------------
            if False:
                if not (UpNeighbor and DownNeighbor and LeftNeighbor and RightNeighbor):
                    print(i.ID,UpNeighbor,DownNeighbor,LeftNeighbor,RightNeighbor)
            if False:
                if not (LeftUpNeighbor and LeftDownNeighbor and RightUpNeighbor and RightDownNeighbor):
                    print(i.ID,LeftUpNeighbor,LeftDownNeighbor,RightUpNeighbor,RightDownNeighbor)
            #----------------------------

            #添加上下左右邻居
            #----------------------------
            if UpNeighbor:
                i.Neighbor.append(AllGrid[i.ID+self.NumGrideWidth])
            if DownNeighbor:
                i.Neighbor.append(AllGrid[i.ID-self.NumGrideWidth])
            if LeftNeighbor:
                i.Neighbor.append(AllGrid[i.ID-1])
            if RightNeighbor:
                i.Neighbor.append(AllGrid[i.ID+1])
            #----------------------------

            #添加左上下右上下邻居
            #----------------------------
            if LeftUpNeighbor:
                i.Neighbor.append(AllGrid[i.ID+self.NumGrideWidth-1])
            if LeftDownNeighbor:
                i.Neighbor.append(AllGrid[i.ID-self.NumGrideWidth-1])
            if RightUpNeighbor:
                i.Neighbor.append(AllGrid[i.ID+self.NumGrideWidth+1])
            if RightDownNeighbor:
                i.Neighbor.append(AllGrid[i.ID-self.NumGrideWidth+1])
            #----------------------------

        for i in AllGrid:
            i.Neighbor.sort(key=lambda Grid: Grid.ID)

        #Debug
        #------------
        if False:
            for i in AllGrid:
                i.Example()
        #------------

        #可视化显示每一个cluster（红色）和他的邻居（随机）
        #----------------------------------------------
        '''
        for i in range(len(AllGrid)):
            print("当前是grid",i,AllGrid[i])
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

        #进行Kmeans聚类，如果有进行过，则不再重新进行
        #----------------------------------------------
        if self.FocusOnLocalRegion == True:
            SaveClusterPath = './data/'+str(self.LocalRegionBound)+str(self.ClustersNumber)+str(self.ClusterMode)+'Label.csv'
        else:
            SaveClusterPath = './data/'+str(self.ClustersNumber)+str(self.ClusterMode)+'Label.csv'
        if os.path.exists(SaveClusterPath):
            ClusterSignal = False
            SaveClusterSignal = False
        else :
            ClusterSignal = True
            SaveClusterSignal = True


        if ClusterSignal :

            if self.ClusterMode == "TransportationClustering":
                self.explog.LogInfo("TransportationClustering in progress")
                estimator = TransportationCluster(
                                                    n_clusters = self.ClustersNumber,
                                                    NodeID = NodeID,
                                                    NodeLocation = NodeLocation,
                                                    NodeIDList = self.NodeIDList,
                                                    Map = ReadNewMap('./data/AccurateMap.csv'),
                                                    NumGrideWidth = self.NumGrideWidth,
                                                    NumGrideHeight = self.NumGrideHeight,
                                                    FocusOnLocalRegion = self.FocusOnLocalRegion,
                                                    LocalRegionBound = self.LocalRegionBound
                                                 )  # 构造聚类器
                estimator.fit()  # 聚类

            elif self.ClusterMode == "TransportationClustering2":
                self.explog.LogInfo("TransportationClustering2 in progress")
                estimator = TransportationCluster2(
                                                    n_clusters = self.ClustersNumber,
                                                    NodeID = NodeID,
                                                    NodeLocation = NodeLocation,
                                                    NodeIDList = self.NodeIDList,
                                                    Map = ReadNewMap('./data/AccurateMap.csv'),
                                                    NumGrideWidth = self.NumGrideWidth,
                                                    NumGrideHeight = self.NumGrideHeight,
                                                    FocusOnLocalRegion = self.FocusOnLocalRegion,
                                                    LocalRegionBound = self.LocalRegionBound
                                                 )  # 构造聚类器
                estimator.fit()  # 聚类

            elif self.ClusterMode == "TransportationClustering3":
                self.explog.LogInfo("TransportationClustering3 in progress")
                estimator = TransportationCluster3(
                                                    n_clusters = self.ClustersNumber,
                                                    NodeID = NodeID,
                                                    NodeLocation = NodeLocation,
                                                    NodeIDList = self.NodeIDList,
                                                    Map = ReadNewMap('./data/AccurateMap.csv'),
                                                    NumGrideWidth = self.NumGrideWidth,
                                                    NumGrideHeight = self.NumGrideHeight,
                                                    FocusOnLocalRegion = self.FocusOnLocalRegion,
                                                    LocalRegionBound = self.LocalRegionBound
                                                 )  # 构造聚类器
                estimator.fit()  # 聚类

            elif self.ClusterMode == "SpectralClustering":
                self.explog.LogInfo("SpectralClustering in progress")
                estimator = SpectralCluster(
                                            n_clusters=self.ClustersNumber,
                                            Knn_k = 5,
                                            NodeID = NodeID,
                                            NodeIDList = self.NodeIDList,
                                            Map = ReadNewMap('./data/AccurateMap.csv'),
                                            #Map = self.Map,
                                            FocusOnLocalRegion = self.FocusOnLocalRegion,
                                            LocalRegionBound = self.LocalRegionBound
                                            )  # 构造聚类器
                estimator.fit()  # 聚类

            elif self.ClusterMode == "OrderClustering":
                self.explog.LogInfo("OrderClustering in progress")
                estimator = OrderCluster(
                                        n_clusters=self.ClustersNumber,
                                        NodeID = NodeID,
                                        NodeLocation = NodeLocation,
                                        NodeIDList = self.NodeIDList,
                                        Orders = ReadOrder('./data/test/order_20161101.csv'),
                                        FocusOnLocalRegion = self.FocusOnLocalRegion,
                                        LocalRegionBound = self.LocalRegionBound
                                        )  # 构造聚类器
                estimator.fit()  # 聚类

            elif self.ClusterMode == "KmeansClustering":
                self.explog.LogInfo("KmeansClustering in progress")
                estimator = KMeans(n_clusters=self.ClustersNumber)  # 构造聚类器
                estimator.fit(NodeLocation)  # 聚类
            
            #estimator = AgglomerativeClustering(linkage='ward', n_clusters=self.ClustersNumber)
            #estimator = DBSCAN(eps=0.0015, min_samples=15)
            
            label_pred = estimator.labels_  # 获取聚类标签

            #label_pred = estimator.ShortestNodeList  # 获取聚类标签

            if SaveClusterSignal :
                label_pred = pd.DataFrame(label_pred)
                label_pred.to_csv(SaveClusterPath,index=0) #不保存列名
                self.explog.LogInfo("Save cluster results to: "+SaveClusterPath)
                reader = pd.read_csv(SaveClusterPath,chunksize = 1000)
                label_pred = []
                for chunk in reader:
                    label_pred.append(chunk)
                label_pred = pd.concat(label_pred)
                label_pred = label_pred.values
                label_pred = label_pred.flatten()
                label_pred = label_pred.astype('int')
        else:
            self.explog.LogInfo("Loading Cluster results")
            reader = pd.read_csv(SaveClusterPath,chunksize = 1000)
            label_pred = []
            for chunk in reader:
                label_pred.append(chunk)
            label_pred = pd.concat(label_pred)
            label_pred = label_pred.values
            label_pred = label_pred.flatten()
            label_pred = label_pred.astype('int')

        #----------------------------------------------

        #将聚类数据载入到实例化的类里
        for i in range(self.ClustersNumber):
            temp = NodeLocation[label_pred == i]
            for j in range(len(temp)):
                Clusters[i].Nodes.append((self.NodeIDList.index(N[(temp[j,0],temp[j,1])]),(temp[j,0],temp[j,1])))

        '''
        for i in Clusters:
            self.PrintCluster(i,True,False)
        plt.show()
        exit()
        '''

        #计算连通性
        #----------------------------------------------
        if self.FocusOnLocalRegion == True:
            SaveClusterTranPath = './data/'+str(self.LocalRegionBound)+str(self.ClustersNumber)+str(self.ClusterMode)+'Tran.csv'
        else:
            SaveClusterTranPath = './data/'+str(self.ClustersNumber)+str(self.ClusterMode)+'Tran.csv'
        if os.path.exists(SaveClusterTranPath):
            TranSignal = False
        else :
            TranSignal = True

        if TranSignal:
            peer2peer = 0
            p2prate = 0

            #建立连通记录矩阵
            tran = [0]*self.ClustersNumber
            for i in range(self.ClustersNumber):
                tran[i] = [0]*self.ClustersNumber

            for i in tqdm(range(self.ClustersNumber)):
                for j in range(len(Clusters[i].Nodes)):
                    #非对称不能用for k in range(i+1,len(Clusters)):
                    for k in range(self.ClustersNumber):
                        for l in range(len(Clusters[k].Nodes)):
                            if self.RoadCost(Clusters[i].Nodes[j][0],Clusters[k].Nodes[l][0]) <= TransportLimtedTimeConnectedThreshold :
                                peer2peer += 1
                        if len(Clusters[k].Nodes) == 0:
                            p2prate = 0
                        else:
                            p2prate = peer2peer/len(Clusters[k].Nodes)
                        if p2prate >= self.p2pConnectedThreshold :
                            tran[Clusters[i].ID][Clusters[k].ID] += 1     #当一个点到另一个grid内所有的点到达率大于0.8时，记为源grid对目标grid的联通点+1
                        peer2peer = 0

            tran = pd.DataFrame(tran)
            tran.to_csv(SaveClusterTranPath,header=0,index=0) #不保存列名

            self.explog.LogInfo("Save the connected record matrix to: "+SaveClusterTranPath)
        else:
            reader = pd.read_csv(SaveClusterTranPath,header = None,chunksize = 1000)
            tran = []
            for chunk in reader:
                tran.append(chunk)
            tran = pd.concat(tran)
            tran = tran.values
            tran = tran.astype('int')
        #----------------------------------------------

        #动态的调整cluster to cluster之间的连通率，从而满足最低邻居要求
        #----------------------------------------------
        g2gConnectedThreshold = 0.99

        for i in range(self.ClustersNumber):
            for j in range(self.ClustersNumber):
                #排除自己是自己邻居的情况
                if i == j:
                    continue

                #当一个grid对另一个grid的联通点，占了自身总点数大于g2gConnectedThreshold，则认为源grid是目标grid的邻居
                if tran[Clusters[i].ID][Clusters[j].ID]/len(Clusters[i].Nodes) >= g2gConnectedThreshold :      
                    Clusters[i].Neighbor.append(Clusters[j])

        #记录不满足邻居数量要求的cluster
        ClusterList = []

        while True:
            if g2gConnectedThreshold <= 0.01:
                break

            #每次都清空列表重新检查
            ClusterList.clear()

            #加入邻居数量未达标的节点
            for i in Clusters:
                if len(i.Neighbor) <= 8:
                    ClusterList.append(i)

            #如果所有节点的邻居都大于3则跳出循环
            if not len(ClusterList):
                break

            #逐级降低连通性要求
            g2gConnectedThreshold -= 0.01

            #降低连通要求重新分配邻居
            for i in ClusterList:
                #先清空之前的邻居分配方案
                i.Neighbor.clear()

                #再对所有节点遍历检查连通性
                for j in Clusters:

                    #排除自己是自己邻居的情况
                    if i == j:
                        continue

                    if tran[i.ID][j.ID]/len(i.Nodes) >= g2gConnectedThreshold:
                        i.Neighbor.append(j)
        #----------------------------------------------


        #----------------------------------------------
        #for i in Clusters:
        #    i.Neighbor = i.Neighbor[:8]
        #----------------------------------------------

        #交叉检查邻居，填补我有他，他没我的情况
        #有向图转无向图
        
        #----------------------------------------------  
        for i in Clusters:
            for j in i.Neighbor:
                if not i in j.Neighbor:
                    j.Neighbor.append(i)
            i.Neighbor.sort(key=lambda Cluster: Cluster.ID)
        #----------------------------------------------

        #self.NodeID2NodesLocation = {}
        for i in Clusters:
            for j in i.Nodes:
                self.NodeID2NodesLocation[j[0]] = j[1]

        #可视化显示每一个cluster（红色）和他的邻居（随机）
        #----------------------------------------------
        '''
        for i in range(len(Clusters)):
            print("当前是grid",i,Clusters[i])
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


    def CreateClusterAdjacencyMatrix(self):
        SaveClusterAdjacencyMatrixPath = './data/'+str(self.LocalRegionBound)+str(self.ClustersNumber)+str(self.ClusterMode)+'AdjacencyMatrix.csv'
        if os.path.exists(SaveClusterAdjacencyMatrixPath):

            reader = pd.read_csv(SaveClusterAdjacencyMatrixPath,header = None,chunksize = 1000)
            ClusterAdjacencyMatrix = []
            for chunk in reader:
                ClusterAdjacencyMatrix.append(chunk)
            ClusterAdjacencyMatrix = pd.concat(ClusterAdjacencyMatrix)
            ClusterAdjacencyMatrix = ClusterAdjacencyMatrix.values
            #ClusterAdjacencyMatrix = ClusterAdjacencyMatrix.astype('int')
        else:
            ClusterAdjacencyMatrix = np.zeros((self.ClustersNumber,self.ClustersNumber)).tolist()
            for i in tqdm(self.Clusters):
                for j in i.Neighbor:
                    temp = round(self.Calculate2ClustersAverageDistance(i,j),2)
                    #if ClusterAdjacencyMatrix[i.ID][j.ID] == 0:
                    ClusterAdjacencyMatrix[i.ID][j.ID] = temp
            ClusterAdjacencyMatrix = np.array(ClusterAdjacencyMatrix)
            ClusterAdjacencyMatrix = pd.DataFrame(ClusterAdjacencyMatrix)
            self.explog.LogInfo("Save the Cluster Adjacency Matrix to: "+SaveClusterAdjacencyMatrixPath)
            ClusterAdjacencyMatrix.to_csv(SaveClusterAdjacencyMatrixPath,header=0,index=0) #不保存列名

        return 


    def Calculate2ClustersAverageDistance(self,cluster1,cluster2):
        TotallyCostFromA2B = 0
        for i in cluster1.Nodes:
            for j in cluster2.Nodes:
                TotallyCostFromA2B += self.RoadCost(i[0],j[0])
        return TotallyCostFromA2B/(len(cluster1.Nodes)+len(cluster2.Nodes))


    def LoadDemandPrediction(self):
        if self.DemandPredictionMode == 'None':
            self.DemandPrediction = None
            return
        elif self.DemandPredictionMode == 'CNN':
            self.DemandPrediction = CNNPredictionModel(PredictionModel_learning_rate = self.PredictionModel_learning_rate,
                                                      NumGrideWidth = self.NumGrideWidth,
                                                      NumGrideHeight = self.NumGrideHeight,
                                                      NumGrideDimension = self.NumGridsDimension,
                                                      OutputDimension = self.PredictionOutputDimension,
                                                      SideLengthMeter = self.SideLengthMeter,
                                                      LocalRegionBound = self.LocalRegionBound,
                                                      train_X = None,
                                                      test_X = None,
                                                      train_Y = None,
                                                      test_Y = None)

            DemandPredictionModelPath = "./model/"+str(self.DemandPredictionMode)+"PredictionModel"+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".h5"

            if os.path.exists(DemandPredictionModelPath):
                self.DemandPrediction.Load(DemandPredictionModelPath)
            else:
                raise Exception("No Demand Prediction Model")

        elif self.DemandPredictionMode == 'INFOCOM':
            self.DemandPrediction = None
        elif self.DemandPredictionMode == 'GCN':
            self.DemandPrediction = None
        else:
            raise Exception('DemandPredictionMode error')

        return


    def Transform2CNNFormat(self,RawData):
        Dimension = len(RawData)
        Res = []

        for i in range(len(RawData[0])):

            TempRes = np.zeros(Dimension)

            for j in range(Dimension):
                TempRes[j] = RawData[j][i]

            Res.append(TempRes)
            
        Res = np.array(Res)

        Res = Res.reshape((self.NumGrideWidth,self.NumGrideHeight,Dimension))

        return Res


    def Transform2GCNFormat(self,RawData):
        Dimension = len(RawData)
        Res = []

        for i in range(len(RawData[0])):

            TempRes = np.zeros(Dimension)

            for j in range(Dimension):
                TempRes[j] = RawData[j][i]

            Res.append(TempRes)

        Res = np.array(Res)

        return Res


    def randomcolor(self):
        colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
        color = ""
        for i in range(6):
            #color += colorArr[random.randint(0,14)]
            color += colorArr[random.randint(0,len(colorArr)-1)]
        return "#"+color


    def PrintCluster(self,Cluster,random=True,show=False):
        randomc = self.randomcolor()
        for i in Cluster.Nodes:
            if random == True:
                plt.scatter(i[1][0],i[1][1],s = 3, c=randomc,alpha = 0.5)
            else :
                plt.scatter(i[1][0],i[1][1],s = 3, c='r',alpha = 0.5)
        if show == True:
            plt.xlim(104.007, 104.13)
            plt.ylim(30.6119, 30.7092)
            plt.show()


    def PrintAllCluster(self):
        for i in self.Clusters:
            self.PrintCluster(i,random=True,show=False)
        plt.xlim(104.007, 104.13)
        plt.ylim(30.6119, 30.7092)
        plt.show()


    def PrintVehicles(self,show = True):
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

        plt.xlim(104.007, 104.13)
        plt.xlabel("red = running  blue = idle  green = rebalance")
        plt.ylim(30.6119, 30.7092)
        if show == True:
            plt.title("Vehicles Location")
            plt.show()


    def PrintVehicleTrajectory(self,Vehicle,show = False):
        X1,Y1 = self.NodeID2NodesLocation[Vehicle.LocationNode]
        X2,Y2 = self.NodeID2NodesLocation[Vehicle.DeliveryPoint]

        #按图片比例放缩
        X1 = (725) / (104.007 - 104.13) * (X1 - 104.13)
        Y1 = (790) / (30.7092 - 30.6119) * (Y1 - 30.6119)

        #起始点
        #plt.scatter(X1,Y1,s = 3, c='black',alpha = 0.3)
        X2 = (725) / (104.007 - 104.13) * (X2 - 104.13)
        Y2 = (790) / (30.7092 - 30.6119) * (Y2 - 30.6119)

        #到达点
        plt.scatter(Y2,X2,s = 3, c='blue',alpha = 0.5)

        LX1=[X1,X2]
        LY1=[Y1,Y2]

        #再平衡的路线
        plt.plot(LY1,LX1,c='k',linewidth=0.3,alpha = 0.5)

        #725*790
        if show == True:
            img=plt.imread('./data/chengdu.png')
            plt.imshow(img)
            plt.title("Vehicles Trajectory")
            #plt.xlim(104.007, 104.13)
            #plt.xlabel("Vehicle Trajectory")
            #plt.ylim(30.6119, 30.7092)
            plt.show()


    def PrintOrder(self,Order):
        
        X1,Y1 = self.NodeID2NodesLocation[Order.PickupPoint]
        #X2,Y2 = self.NodeID2NodesLocation[Order.DeliveryPoint]

        if False:
        #按图片比例放缩
            X1 = (725) / (104.007 - 104.13) * (X1 - 104.13)
            Y1 = (790) / (30.7092 - 30.6119) * (Y1 - 30.6119)

        #起始点
        plt.scatter(X1,Y1,s = 3, c='r',alpha = 0.5)

        '''
        X2 = (725) / (104.007 - 104.13) * (X2 - 104.13)
        Y2 = (790) / (30.7092 - 30.6119) * (Y2 - 30.6119)
        #到达点
        plt.scatter(X2,Y2,s = 3, c='blue',alpha = 0.5)
        '''


    #bootrap fuction
    def CreateOrderDatabase(self,OriginalOrders):
        N = len(OriginalOrders)

        res = []
        for i in range(N):
            RandomChoice = random.randint(0,N-1)
            res.append(OriginalOrders[RandomChoice].tolist())

        res = sorted(res, key = lambda x:x[0])
        res = np.array(res)

        return res


    def WorkdayOrWeekend(self,day):
        if type(day) != type(0) or day<0 or day > 6:
            raise Exception('input format error')
        elif day == 5 or day == 6:
            return "Weekend"
        else:
            return "Workday"


    def timestamp2vec(self,timestamps):
        # tm_wday range [0, 6], Monday is 0
        vec = [time.strptime(str(t[:6])).tm_wday for t in timestamps]  # python3
        ret = []
        for i in vec:
            v = [0 for _ in range(7)]
            v[i] = 1
            if i >= 5: 
                v.append(0)  # weekend
            else:
                v.append(1)  # weekday
            ret.append(v)
        return ret


    def GetTimeWeatherState(self,Order):
        #Year = Order.ReleasTime.year
        Month = Order.ReleasTime.month
        Day = Order.ReleasTime.day
        #0=周一 6=周日
        Week = Order.ReleasTime.weekday()
        Hour = Order.ReleasTime.hour
        Minute = Order.ReleasTime.minute

        if Month == 11:
            if Hour < 12:
                Weather = self.WeatherTable11[2*(Day-1)]
            else:
                Weather = self.WeatherTable11[2*(Day-1)+1]
        elif Month == 12:
            if Hour < 12:
                Weather = self.WeatherTable12[2*(Day-1)]
            else:
                Weather = self.WeatherTable12[2*(Day-1)+1]
        else:
            raise Exception('Month format error')

        #TimeState = [Year,Month,Day,Week,Hour,Minute]
        #TimeState = [Day/31,Week/7,Hour/24,Minute/60,Weather/4]
        TimeState = [Day,Week,Hour,Minute,Weather]

        return TimeState


    def GetOneHot(self,N,Len=144):
        ResultOneHot = np.zeros(Len)
        ResultOneHot[N] = 1.0
        ResultOneHot = ResultOneHot.tolist()
        return ResultOneHot


    def GetTimeWeatherOneHotNormalizationState(self,Order):
        Month = Order.ReleasTime.month
        Day = Order.ReleasTime.day
        #0=周一 6=周日
        Week = Order.ReleasTime.weekday()
        Hour = Order.ReleasTime.hour
        Minute = Order.ReleasTime.minute

        if Month == 11:
            if Hour < 12:
                Weather = self.WeatherTable11[2*(Day-1)]
            else:
                Weather = self.WeatherTable11[2*(Day-1)+1]
        elif Month == 12:
            if Hour < 12:
                Weather = self.WeatherTable12[2*(Day-1)]
            else:
                Weather = self.WeatherTable12[2*(Day-1)+1]

        #to one-hot
        TimeState = []
        TimeState += self.GetOneHot(Weather,4)
        TimeState += self.GetOneHot(Day-1,31)
        TimeState += self.GetOneHot(Week-1,7)
        TimeState += self.GetOneHot((Hour*6 + Minute//10),1440//TIMESTEPNUMBER)

        return TimeState


    #读取记忆文件夹得到记忆的位置
    def ReadSaveNum(self,filepath = "./remember/"):
        RememberList = os.listdir(filepath)
        RememberIntList = []
        for i in RememberList:
            temp = re.findall(r'_\d+_', i)
            if len(temp):
                RememberIntList.append(temp[0])
        RememberNewIntList = []
        if len(RememberIntList):
            for i in RememberIntList:
                i = i[1:-1]
                i = int(i)
                RememberNewIntList.append(i)
            RememberNewIntList.sort()
            self.SaveNumber = RememberNewIntList[-1] + 1
        else:
            self.SaveNumber = 0

        return 


    #---------------------------------------------------------------------------
    def DemandPredictFunction(self):

        return


    def RebalanceFunction(self):

        return


    def SimulationFunction(self):
        #匹配订单
        #------------------------------------------------
        #当实验时间经过一个时间片
        while self.NowOrder.ReleasTime < self.RealExpTime+self.TimePeriods :

            if self.NowOrder.ID == self.Orders[-1].ID:
                break

            self.OrderNum += 1

            NowCluster = self.NodeID2Cluseter[self.NowOrder.PickupPoint]

            NowCluster.Orders.append(self.NowOrder)

            if len(NowCluster.IdleVehicles) or len(NowCluster.Neighbor):

                TempMin = None

                if len(NowCluster.IdleVehicles):

                    #找最近的车给当前order
                    #--------------------------------------
                    #TempMin = None
                    for i in NowCluster.IdleVehicles:

                        TempRoadCost = self.RoadCost(i.LocationNode,self.NowOrder.PickupPoint)

                        #得到遍历到的当前车到接客点的距离耗费
                        if TempMin == None :
                            TempMin = (i,TempRoadCost,NowCluster)
                        elif TempRoadCost < TempMin[1] :
                            TempMin = (i,TempRoadCost,NowCluster)
                    #--------------------------------------

                #邻居找车系统，增大搜索范围
                elif self.NeighborCanServer and len(NowCluster.Neighbor):

                    TempMin = self.FindServerVehicleFunction(
                                                            NeighborServerDeepLimit=self.NeighborServerDeepLimit,
                                                            Visitlist={},
                                                            Cluster=NowCluster,
                                                            TempMin=None,
                                                            deep=0
                                                            )

                #This means all Neighbor Cluster have no IdleVehicles
                if TempMin == None or TempMin[1] > PICKUPTIMEWINDOW:
                    self.RejectNum+=1
                    self.NowOrder.ArriveInfo="Reject"
                #Succ
                else:
                    NowVehicle = TempMin[0]
                    self.NowOrder.PickupWaitTime = TempMin[1]
                    NowVehicle.Orders.append(self.NowOrder)

                    self.TotallyWaitTime += self.RoadCost(NowVehicle.LocationNode,self.NowOrder.PickupPoint)

                    ScheduleCost = self.RoadCost(NowVehicle.LocationNode,self.NowOrder.PickupPoint) + self.RoadCost(self.NowOrder.PickupPoint,self.NowOrder.DeliveryPoint)

                    NowVehicle.DeliveryPoint = self.NowOrder.DeliveryPoint

                    #Delivery Cluster {Vehicle:ArriveTime}
                    self.Clusters[self.NodeID2Cluseter[self.NowOrder.DeliveryPoint].ID].VehiclesArrivetime[NowVehicle] = self.RealExpTime + np.timedelta64(ScheduleCost*MINUTES)

                    #delete now Cluster's recode about now Vehicle
                    TempMin[2].IdleVehicles.remove(NowVehicle)

                    self.NowOrder.ArriveInfo="Success"
            else:
                #该cluster内没有一辆可用空闲车辆
                self.RejectNum += 1    
                self.NowOrder.ArriveInfo = "Reject"

            #进行完一个订单，读取下一个订单
            #------------------------------
            self.NowOrder = self.Orders[self.NowOrder.ID+1]
        #------------------------------------------------

        #update Demand prediction input orders data
        #------------------------------------------------
        self.LastThreeStepOrdersMap = self.LastTwoStepOrdersMap.copy()
        self.LastTwoStepOrdersMap = self.LastOneStepOrdersMap.copy()
        for i in range(self.ClustersNumber):
            self.LastOneStepOrdersMap[i] = len(self.Clusters[i].Orders)

        #print(self.LastOneStepOrdersMap)
        #print(self.LastTwoStepOrdersMap)
        #exit()

        return


    #dfs访问自身和邻居，找到最近的车
    def FindServerVehicleFunction(self,NeighborServerDeepLimit,Visitlist,Cluster,TempMin,deep):
        #找临近的Cluster内的IdleVehicles
        if deep > NeighborServerDeepLimit or Cluster.ID in Visitlist:
            return TempMin

        Visitlist[Cluster.ID] = True
        for i in Cluster.IdleVehicles:
            TempRoadCost = self.RoadCost(i.LocationNode,self.NowOrder.PickupPoint)
            if TempMin == None :
                TempMin = (i,TempRoadCost,Cluster)
            elif TempRoadCost < TempMin[1]:
                TempMin = (i,TempRoadCost,Cluster)
        #acc
        if self.NeighborCanServer:
            for j in Cluster.Neighbor:
                TempMin = self.FindServerVehicleFunction(NeighborServerDeepLimit,Visitlist,j,TempMin,deep+1)
        return TempMin


    def RewardFunction(self):
        return


    def UpdateFunction(self):
        #更新
        #------------------------------------------------
        for i in self.Clusters:

            #根据到达时间清空到达表
            i.RebalanceNumber = 0

            for key,value in list(i.VehiclesArrivetime.items()):
                #key = Vehicle ; value = Arrivetime
                if value <= self.RealExpTime :

                    #update Order
                    if len(key.Orders):
                        key.Orders[0].ArriveOrderTimeRecord(self.RealExpTime)
                        #print(key.Orders[0].ID,key.Orders[0].ArriveInfo)

                    #update Vehicle info
                    #key.ArriveVehicleUpDate(key.Orders[0].DeliveryPoint, i)
                    key.ArriveVehicleUpDate(i)

                    #update Cluster record
                    i.ArriveClusterUpDate(key)
        #------------------------------------------------

        return


    def GetState_Function(self):
        return


    def LearningFunction(self):
        return


    def RememberFunction(self):
        if SaveMemorySignal == True :
            if len(self.SaveMemory) > 20000 :
                CSVSaver = pd.DataFrame(self.SaveMemory)
                self.SaveMemory.clear()
                CSVSaver.to_csv("./remember/new/" + str(self.ClustersNumber) + 'Cluster' + str(self.VehiclesNumber) + 'Vehicles_' + str(self.SaveNumber) + '_Exp.csv',header=0,index=0,mode='a')
                del CSVSaver
        return


    def SimCity(self,SaveMemorySignal = False):

        self.RealExpTime = self.Orders[0].ReleasTime
        EndTime = self.Orders[-1].ReleasTime + 3 * self.TimePeriods
        self.NowOrder = self.Orders[0]
        self.step = 0


        #self.PrintAllCluster()

        #PrintVehicles(self.Clusters)

        EpisodeStartTime = dt.datetime.now()

        #self.explog.LogInfo("开始实验")

        while self.RealExpTime <= EndTime:

            LastRebalanceNum = self.RebalanceNum

            LastRejectNum = self.RejectNum

            LastOrderNum = self.OrderNum

            StepStartTime = dt.datetime.now()

            #PrintVehicles(self.Clusters)
            '''
            if self.RealExpTime.hour == 12 and self.RealExpTime.minute < 10 :
                PrintVehicles(self.Clusters)
            if self.RealExpTime.hour == 23 and self.RealExpTime.minute < 10 :
                PrintVehicles(self.Clusters)
            '''

            '''
            img=plt.imread('./data/chengdu.png')
            plt.imshow(img)

            #725*790

            #plt.xlim(104.007, 104.13)
            plt.xlabel("Vehicle Trajectory")
            #plt.ylim(30.6119, 30.7092)

            plt.show()
            '''
            StepDemandPredictStartTime = dt.datetime.now()

            self.DemandPredictFunction()

            self.TotallyDemandPredictTime += dt.datetime.now() - StepDemandPredictStartTime  


            #计算匹配前的空车数量
            #------------------------------------------------
            for i in self.Clusters:
                i.PerRebalanceIdleVehicles = len(i.IdleVehicles)
            #------------------------------------------------

            #记录每一步再平衡的开始时间
            StepRebalanceStartTime = dt.datetime.now()

            self.RebalanceFunction()

            #记录每一步再平衡的结束时间
            self.TotallyRebalanceTime += dt.datetime.now() - StepRebalanceStartTime  

            for i in self.Clusters:
                i.Orders.clear()

            #记录开始模拟匹配的时间
            StepSimulationStartTime = dt.datetime.now()

            #计算匹配前的空车数量
            #------------------------------------------------
            for i in self.Clusters:
                #i.TempIdleVehicles = len(i.IdleVehicles)
                i.PerMatchIdleVehicles = len(i.IdleVehicles)
            #------------------------------------------------

            self.SimulationFunction()

            self.RewardFunction()

            #记录结束模拟匹配和计算发放奖励的时间
            self.TotallySimulationTime += dt.datetime.now() - StepSimulationStartTime


            #记录每一步更新的开始时间
            StepUpdateStartTime = dt.datetime.now()

            self.UpdateFunction()

            self.GetState_Function()

            self.TotallyUpdateTime += dt.datetime.now() - StepUpdateStartTime


            StepRememberStartTime = dt.datetime.now()

            self.RememberFunction()

            self.TotallyRememberTime += dt.datetime.now() - StepRememberStartTime

            
            #记录每一步学习的开始时间
            StepLearningStartTime = dt.datetime.now()

            self.LearningFunction()

            #记录每一步学习的结束时间
            self.TotallyLearningTime += dt.datetime.now() - StepLearningStartTime


            #print("StepReward:", self.StepReward)
            self.TotallyReward += self.StepReward
            self.StepReward = 0


            #计算载客率
            Working = 0
            Idle = 0
            if self.NowOrder.ID != self.Orders[-1].ID:
                for j in self.Vehicles:
                    if len(j.Orders):
                        Working += 1
                    else:
                        Idle += 1

            self.SumWorking += Working
            self.SumIdle += Idle


            self.step += 1

            self.RealExpTime += self.TimePeriods

            StepEndTime = dt.datetime.now()

            '''
            print("RealExpTime:", self.RealExpTime , " StepRunTime:" , StepEndTime - StepStartTime)
            #self.explog.LogInfo("RealExpTime:" + str(self.RealExpTime) + "Step Run Time : " + str(StepEndTime - StepStartTime))

            print("StepOrders:",self.OrderNum - LastOrderNum," RebalanceNumber:",self.RebalanceNum - LastRebalanceNum ," Reject Number:",self.RejectNum - LastRejectNum," AgentTempPool:",len(self.AgentTempPool))

            if Working != 0:
                print("载客率:",round(100*Working/(Working+Idle),4),"%")
                pass
            else:
                print("载客率:0%")
            print()
            '''

        ##------------------------------------------------


        EpisodeEndTime = dt.datetime.now()


        if SaveMemorySignal == True :
            CSVSaver = pd.DataFrame(self.SaveMemory)
            self.SaveMemory.clear()
            CSVSaver.to_csv("./remember/new/" + str(self.ClustersNumber) + 'Cluster' + str(self.VehiclesNumber) + 'Vehicles_' + str(self.SaveNumber) + '_Exp.csv',header=0,index=0,mode='a')
            del CSVSaver


        SumOrderValue = 0
        OrderValueNum = 0

        for i in self.Orders:
            if i.ArriveInfo != "Reject":
                SumOrderValue += i.OrderValue
                OrderValueNum += 1


        self.explog.LogInfo("实验结束")
        self.explog.LogInfo("Cluster's mode: " + self.ClusterMode)
        self.explog.LogInfo("Rebalance's mode: " + self.RebalanceMode)
        self.explog.LogInfo("Neighbor can server?: " + str(self.NeighborCanServer))
        self.explog.LogInfo("Date: " + str(self.Orders[0].ReleasTime.month) + "/" + str(self.Orders[0].ReleasTime.day))
        self.explog.LogInfo("Weekend or Workday: " + self.WorkdayOrWeekend(self.Orders[0].ReleasTime.weekday()))
        self.explog.LogInfo("p2p Connected Threshold is: " + str(self.p2pConnectedThreshold))
        if self.ClusterMode != "Grid":
            self.explog.LogInfo("Number of Clusters: " + str(len(self.Clusters)))
        elif self.ClusterMode == "Grid":
            self.explog.LogInfo("Number of Grids: " + str((self.NumGrideWidth * self.NumGrideHeight)))
        self.explog.LogInfo("Number of Vehicles: " + str(len(self.Vehicles)))
        self.explog.LogInfo("Number of Orders: " + str(len(self.Orders)))
        self.explog.LogInfo("Number of Reject: " + str(self.RejectNum))
        if (len(self.Orders)-self.RejectNum)!=0:
            self.explog.LogInfo("Average wait time: " + str(self.TotallyWaitTime/(len(self.Orders)-self.RejectNum)))
        self.explog.LogInfo("Number of Rebalance: " + str(self.RebalanceNum))
        if (self.RebalanceNum)!=0:
            self.explog.LogInfo("Average Rebalance Cost: " + str(self.TotallyRebalanceCost/self.RebalanceNum))
        if (self.SumWorking+self.SumIdle)!=0:
            self.explog.LogInfo("载客率: " + str(round(100*self.SumWorking/(self.SumWorking+self.SumIdle),4)) + "%")
        self.explog.LogInfo("Totally Order value: " + str(SumOrderValue))
        if OrderValueNum!=0:
            self.explog.LogInfo("Number of each Servers Order value: " + str(round(SumOrderValue/OrderValueNum,3)))
        self.explog.LogInfo("Totally Reward : " + str(self.TotallyReward))
        self.explog.LogInfo("Totally Update Time : " + str(self.TotallyUpdateTime))
        self.explog.LogInfo("Totally Remember Time : " + str(self.TotallyRememberTime))
        self.explog.LogInfo("Totally Learning Time : " + str(self.TotallyLearningTime))
        self.explog.LogInfo("Totally Demand Predict Time : " + str(self.TotallyDemandPredictTime))
        self.explog.LogInfo("Totally Rebalance Time : " + str(self.TotallyRebalanceTime))
        self.explog.LogInfo("Totally Simulation Time : " + str(self.TotallySimulationTime))
        self.explog.LogInfo("Episode Run time : " + str(EpisodeEndTime - EpisodeStartTime))

        '''
        print('----------------------------------------------------------------')
        print()

        
        for i in dir():
            print(i,sys.getsizeof(i))

        print('----------------------------------------------------------------')
        print()
        '''

        return

