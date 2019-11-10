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
#import multiprocessing as mp
from collections import deque
from objects.objects import Cluster,Order,Vehicle,Agent,Grid
from config.setting import *
from preprocessing.readfiles import *
from tools.tools import *
from dqn.KerasAgent import DQNAgent
from simulator.simulator import Logger,Simulation

from tqdm import tqdm
from matplotlib.pyplot import plot,savefig
from sklearn.cluster import KMeans
#数组转one hot用
#from keras.utils import to_categorical

#from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,AffinityPropagation
###########################################################################

class ContextGrid(Grid):

    def __init__(self,ID,Nodes,Neighbor,NeighborArriveList,IdleVehicles,VehiclesArrivetime,Orders):
        self.ID = ID
        self.Nodes = Nodes
        self.Neighbor = Neighbor
        self.NeighborArriveList = NeighborArriveList    #{v1:arrivetime,....}//{t1:[v:node],t2:[v:node]...}
        self.IdleVehicles = IdleVehicles
        #self.SpatialList = SpatialList #{grid:TravelCostDistance,grid:TravelCostDistance...}
        self.VehiclesArrivetime = VehiclesArrivetime
        self.Orders = Orders
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0

        self.GeographicalContext = None


    def GetClusterState(self,ClustersNumber):
        #ohl = keras.utils.to_categorical(self.ID,num_classes = ClustersNumber)
        IDState = []
        for i in range(ClustersNumber):
            if self.ID == i:
                IDState.append(1)
            else:
                IDState.append(0)
        
        GridState = [len(self.IdleVehicles),len(self.Orders)] + IDState

        return GridState


class Simulation(Simulation):

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

        AllGrid = [ContextGrid(i,[],[],{},[],{},[]) for i in range(NumGride)]

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
                print("error")
                print(key[0],key[1])
                exit()
            else:
                AllGrid[self.NumGrideWidth * NowGridHeightNum + NowGridWidthNum].Nodes.append((value,(key[0],key[1])))
        #------------------------------------------------------


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

        for i in AllGrid:
            for j in i.Nodes:
                self.NodeID2NodesLocation[j[0]] = j[1]


        #添加GeographicalContext
        #----------------------
        for i in AllGrid:
            if len(i.Nodes):
                i.GeographicalContext = 1
            else:
                i.GeographicalContext = 0
        #----------------------

        return AllGrid



    def GetTimeState(self,Order):
        #Year = Order.ReleasTime.year
        #Month = Order.ReleasTime.month
        Day = Order.ReleasTime.day / 30
        #0=周一 6=周日
        Week = Order.ReleasTime.weekday() / 7
        Hour = Order.ReleasTime.hour / 24
        Minute = Order.ReleasTime.minute / 60

        #TimeState = [Year,Month,Day,Week,Hour,Minute]
        TimeState = [Day,Week,Hour,Minute]

        return TimeState


    def RewardFunction(self):
        #计算奖励
        #------------------------------------------------
        for i in self.Clusters:

            if self.RealExpTime > self.Orders[-1].ReleasTime:
                for j in self.AgentTempPool:
                    #如果正在调度，还没有到，没得到奖励则跳过
                    j.PositiveReward = 0
                break

            #统计每个Cluster每轮的订单数量
            StepOrdersSumValue = 0

            for j in i.Orders:
                StepOrdersSumValue += j.OrderValue

            #清除上一次Cluster内所有出现AllClusters的订单的记录
            #i.Orders.clear()

            #PositiveReward
            AllClusterPositiveReward = None
            if i.PerMatchIdleVehicles == 0:
                AllClusterPositiveReward = 0
            else:
                AllClusterPositiveReward = StepOrdersSumValue / i.PerMatchIdleVehicles
            #------------------

            #给Agent发放奖励
            for j in self.AgentTempPool:
                #正面奖励
                if AllClusterPositiveReward != None:
                    if j.ArriveCluster == i and j.Vehicle.Cluster == i:
                        if j.PositiveReward != None:
                            j.PositiveReward += PositiveRewardHyperparameter * AllClusterPositiveReward
                        else:
                            j.PositiveReward = PositiveRewardHyperparameter * AllClusterPositiveReward

        #------------------------------------------------

        return


    def GetState_Function(self):
        #结算上一轮
        #---------------------------------------------
        # j = Agent
        for j in self.AgentTempPool:

            #如果正在调度，还没有到，没得到奖励则跳过
            if j.PositiveReward == None :
                continue

            j.TotallyReward = j.PositiveReward

            #得到新的Cluster的State情况
            ClusterState_ = j.FromCluster.GetClusterState(self.ClustersNumber)

            TimeState_ = self.GetTimeState(self.NowOrder)

            j.State_ = TimeState_ + ClusterState_

            j.State_ = np.reshape(j.State_, [1, DQN.state_size])

            DQN.remember(j.State, j.Action, j.TotallyReward, j.State_, False)

            if SaveMemorySignal == True :
                self.SaveMemory.append((j.State, j.Action, j.TotallyReward, j.State_))

        for i in range(len(self.AgentTempPool)-1, -1, -1):
            #删除缓存池里已经完整的经验，对还没有完成旅程的经验不处理
            if len(self.AgentTempPool[i].State_):
                self.AgentTempPool.pop(i)

        return


    def RebalanceFunction(self):
        #Policy
        #------------------------------------------------
        TimeState = self.GetTimeState(self.NowOrder)

        for i in self.Clusters:

            if self.RealExpTime > self.Orders[-1].ReleasTime:
                break

            if i.GeographicalContext == 0:
                continue
            
            ClusterState = i.GetClusterState(self.ClustersNumber)

            State = TimeState + ClusterState

            State = np.reshape(State, [1, DQN.state_size])
            
            QValue = 0
            while QValue == 0:
                #Action = DQN.action(State,actionnum = len(i.Neighbor))

                #print(Action,len(i.Neighbor))
                act_values = DQN.model.predict(State)
                Actionset = act_values[0]
                
                if np.random.rand() <= DQN.epsilon:
                    Action = random.randrange(len(i.Neighbor))
                else:
                    Action = np.argmax(Actionset)


                if Action-1 >= len(i.Neighbor):
                    continue

                #get Collaborative Context
                if Actionset[0] >= Actionset[Action] :
                    CollaborativeContext = 1
                else:
                    CollaborativeContext = 0

                if Action == 0:
                    #print(1,Actionset[Action],CollaborativeContext,i.GeographicalContext)
                    QValue = Actionset[Action] * CollaborativeContext * i.GeographicalContext
                else:
                    #print(2,Actionset[Action],CollaborativeContext,i.Neighbor[Action-1].GeographicalContext)
                    QValue = Actionset[Action] * CollaborativeContext * i.Neighbor[Action-1].GeographicalContext


            #j = each IdleVehicles in each Cluster
            for j in i.IdleVehicles:

                #建立Agent实例
                NowAgent = Agent(
                                FromCluster = i,
                                ArriveCluster = None,
                                Vehicle = j,
                                State = State,
                                Action = Action,
                                TotallyReward = None,
                                PositiveReward = None,
                                NegativeReward = None,
                                NeighborNegativeReward = None,
                                State_ = []
                                )

                self.AgentTempPool.append(NowAgent)

                #调度操作
                #随机抽到达点
                #------------------
                if Action > len(i.Neighbor):
                    NowAgent.PositiveReward = -RewardValue
                    #非法调度，停留原地
                    NowAgent.ArriveCluster = NowAgent.FromCluster




                #当Action是0时，指向停留在原Cluster
                elif Action == 0:
                    #explog.LogDebug("停留在原地")
                    NowAgent.ArriveCluster = NowAgent.FromCluster
                    #停留在原地没有惩罚


                #Action < len(i.Neighbor):
                else:
                    #ArriveCluster = i.Neighbor[Action]
                    ArriveCluster = i.Neighbor[Action-1]
                    NowAgent.ArriveCluster = ArriveCluster

                    #从限定时间内的到达点（10min）里随机选择
                    if False:
                        TempCostList = []

                        while not len(TempCostList):
                            for k in range(len(ArriveCluster.Nodes)):
                                DeliveryPoint = ArriveCluster.Nodes[k][0]
                                if self.RoadCost(j.LocationNode,DeliveryPoint) < RebalanceTimeLim:
                                    TempCostList.append(DeliveryPoint)

                        DeliveryPoint = random.choice(TempCostList)
                        #调度惩罚
                        #------------------------------
                        if NowAgent.NegativeReward == None:
                            NowAgent.NegativeReward = -0.6
                        else:
                            NowAgent.NegativeReward += -0.6
                        #------------------------------
                        j.DeliveryPoint = DeliveryPoint

                    #直接重定位到离当前最近的到达点
                    elif False:
                        mostlow = ArriveCluster.Nodes[0][0]

                        for k in range(len(ArriveCluster.Nodes)):
                            DeliveryPoint = ArriveCluster.Nodes[k][0]
                            #j.DeliveryPoint = DeliveryPoint
                            if self.RoadCost(j.LocationNode,DeliveryPoint) < mostlow:
                                mostlow = self.RoadCost(j.LocationNode,DeliveryPoint)
                                j.DeliveryPoint = DeliveryPoint

                    #在最短路径的到达点集合里随机选择
                    elif False:
                        TempCostList = {}
                        for k in ArriveCluster.Nodes:
                            TempCostList[k[0]] = self.RoadCost(j.LocationNode,k[0])

                        TempCostList = sorted(TempCostList,key=TempCostList.__getitem__)

                        if len(TempCostList) <= 5 and len(TempCostList) > 2:
                            j.DeliveryPoint = random.choice(TempCostList[:4])
                        elif len(TempCostList) <= 2 :
                            j.DeliveryPoint = TempCostList[2]
                        else :
                            j.DeliveryPoint = random.choice(TempCostList[:6])

                    #在所有到达点里随机选择
                    elif True:
                        j.DeliveryPoint = random.choice(ArriveCluster.Nodes)[0]

                    #Delivery Cluster {Vehicle:ArriveTime}
                    ArriveCluster.VehiclesArrivetime[j] = self.RealExpTime + np.timedelta64(self.RoadCost(j.LocationNode,j.DeliveryPoint)*MINUTES)

                    #delete now Cluster's recode about now Vehicle
                    i.IdleVehicles.remove(j)   

                    self.RebalanceNum += 1

        #------------------------------------------------
        for i in self.Clusters:
            #清除上一次Cluster内所有出现AllClusters的订单的记录
            i.Orders.clear()
        #------------------------------------------------

        return


    def LearningFunction(self):
        #learning
        #------------------------------------------------
        #if len(DQN.memory) > DQN.batch_size:
        if (len(DQN.memory) > DQN.batch_size) and (self.step % 12 == 0):

            for i in range(4096//DQN.batch_size):
                ExpLossHistory = DQN.replay(DQN.batch_size)

            if DQN.epsilon > DQN.epsilon_min:
                DQN.epsilon *= DQN.epsilon_decay

            print("loss均值: ",round(np.mean(ExpLossHistory),5),"loss方差: ",round(np.var(ExpLossHistory),5),"epsilon: ",round(DQN.epsilon,5))
        #------------------------------------------------ 

        #每x次step更换参数
        #---------------------------------------------
        #if (self.step > 30) and (self.step % 10 == 0):
        if (self.Episode % 3 == 0):
            DQN.update_target_model()
        #---------------------------------------------


        return



if __name__ == "__main__":
    


    explog = Logger()

    p2pConnectedThreshold = 0.8
    ClusterMode = "Grid"
    RebalanceMode = "KDD"
    ClustersNumber = 30
    NumGrideWidth = 6
    NumGrideHeight = 5
    VehiclesNumber = 2000
    #TIMESTEP = np.timedelta64(10*MINUTES)
    NeighborCanServer = False
    NeighborServerDeepLimit = 3
    FocusOnLocalRegion = True
    LocalRegionBound = (104.045,104.095,30.635,30.685)


    EXPSIM = Simulation(
                        explog = explog,
                        p2pConnectedThreshold = p2pConnectedThreshold,
                        ClusterMode = ClusterMode,
                        RebalanceMode = RebalanceMode,
                        ClustersNumber = ClustersNumber,
                        NumGrideWidth = NumGrideWidth,
                        NumGrideHeight = NumGrideHeight,
                        VehiclesNumber = VehiclesNumber,
                        TimePeriods = TIMESTEP,
                        NeighborCanServer = NeighborCanServer,
                        NeighborServerDeepLimit = NeighborServerDeepLimit,
                        FocusOnLocalRegion = FocusOnLocalRegion,
                        LocalRegionBound = LocalRegionBound
                        )


    EXPSIM.CreateAllInstantiate()


    action_size = 9
    state_size = 2 + 4 + EXPSIM.ClustersNumber

    print("state_size",state_size,"action_size",action_size)

    DQN = DQNAgent(
                    state_size = state_size,
                    action_size = action_size,
                    #memory_size = deque(maxlen=4096),
                    memory_size = deque(maxlen=20000),
                    gamma = 0.95,
                    epsilon = 0.40,
                    epsilon_min = 0.01,
                    epsilon_decay = 0.99,
                    learning_rate = 0.001,
                    batch_size = 32
                    )


    #DQN.load("./model/kdd.h5")

    OrderFileDate = ["1101"]

    while EXPSIM.Episode < 500:
        if EXPSIM.Episode == 100:
            DQN.learning_rate = 0.0001

        if SaveMemorySignal == True :
            EXPSIM.ReadSaveNum(filepath = "./remember/new")
        EXPSIM.SimCity(SaveMemorySignal)

        DQN.save("./model/kdd.h5")

        if SaveMemorySignal == True :
            EXPSIM.SaveNumber= EXPSIM.SaveNumber + 1

        EXPSIM.Reset(OrderFileDate[0])

        EXPSIM.Episode += 1
