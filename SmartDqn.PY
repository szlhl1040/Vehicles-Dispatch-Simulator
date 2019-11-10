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
from simulator.simulator import Logger,Simulation
#from simulator.CNNDeamdPrediction import CNNPredictionModel

from collections import deque
from objects.objects import Cluster,Order,Vehicle,Agent,Grid
from config.setting import *
#from preprocessing.readfiles import *
from dqn.CNNAgent import CNNDQNAgent
from tqdm import tqdm
from matplotlib.pyplot import plot,savefig
from sklearn.cluster import KMeans
#数组转one hot用
#from keras.utils import to_categorical

#from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,AffinityPropagation

###########################################################################


class SmartSimulation(Simulation):

    def DefiningLocalVariables(self):

        return


    def GetStateFunction(self,NowCluster):
        IntervalSize = 10

        NowClusterState = NowCluster.Int2BinList(NowCluster.ID)
        NowClusterState += [len(NowCluster.IdleVehicles)//IntervalSize,self.DeamdPredictionOutput[NowCluster.ID]]

        NeighborClusterState = []
        for i in NowCluster.Neighbor:
            NeighborClusterState += [len(i.IdleVehicles)//IntervalSize,self.DeamdPredictionOutput[i.ID]]

        State = NowClusterState + NeighborClusterState

        #fill State use ZERO
        while (True) :
            if len(State) < DQN.state_size :
                #当ClusterState不足时，用0填充
                State.append(0)
            elif len(State) == DQN.state_size:
                break
            else :
                raise Exception('Cluster State Size Error!')

        return State


    #---------------------------------------------------------------------------
    def RebalanceFunction(self):
        #Policy
        #------------------------------------------------

        #DeamdPrediction
        #-------------------------
        TimeWeatherState = self.GetTimeWeatherState(self.NowOrder)

        #TimeWeatherState[1] = day of week   
        self.SinDayOfWeek.fill(np.sin(TimeWeatherState[1]))
        self.CosDayOfWeek.fill(np.cos(TimeWeatherState[1]))
        #TimeWeatherState[2] = hour of day
        self.SinHourOfDay.fill(np.sin(TimeWeatherState[2]))
        self.CosHourOfDay.fill(np.cos(TimeWeatherState[2]))
        #TimeWeatherState[2] = minute of hour
        self.SinMinuteOfHour.fill(np.sin(TimeWeatherState[3]))
        self.CosMinuteOfHour.fill(np.cos(TimeWeatherState[3]))
        #Weather
        self.Weather.fill(TimeWeatherState[4]/4.0)

        '''
        print(self.LastTwoStepOrdersMap)
        print(self.LastOneStepOrdersMap)
        print(self.SinDayOfWeek)
        print(self.CosDayOfWeek)
        print(self.SinHourOfDay)
        print(self.CosHourOfDay)
        print(self.SinMinuteOfHour)
        print(self.CosMinuteOfHour)
        print(self.Weather)
        '''

        self.DeamdPredictionInput = self.Transform2CNNFormat([self.LastTwoStepOrdersMap,
                                                              self.LastOneStepOrdersMap,
                                                              self.SinDayOfWeek,
                                                              self.CosDayOfWeek,
                                                              self.SinHourOfDay,
                                                              self.CosHourOfDay,
                                                              self.SinMinuteOfHour,
                                                              self.CosMinuteOfHour,
                                                              self.Weather])

        self.DeamdPredictionInput = np.array([self.DeamdPredictionInput])
        


        self.DeamdPredictionOutput = self.DeamdPrediction.predict(self.DeamdPredictionInput)
        self.DeamdPredictionOutput = self.DeamdPredictionOutput[0]
        
        print("-----input----------------")
        print(self.DeamdPredictionInput)
        print(self.DeamdPredictionInput.shape)
        print("-----output----------------")
        print(self.DeamdPredictionOutput)
        print(self.DeamdPredictionOutput.shape)
        print("---------------------------")
        
        #-------------------------

        TempAllClusters = self.Clusters[:]
        random.shuffle(TempAllClusters)

        #i = each Cluster
        for i in TempAllClusters:

            if self.RealExpTime > self.Orders[-1].ReleasTime:
                break

            #j = each IdleVehicles in each Cluster
            for j in i.IdleVehicles:

                #ClusterState = i.GetClusterIdleVehiclesState(self.state_size)

                State = self.GetStateFunction(NowCluster=i)

                State = np.reshape(State, [1, DQN.state_size])


                #最为费时
                if False:
                #if DQN.epsilon >= 0.05:
                    Action = DQN.action(State,actionnum = len(i.Neighbor))
                else:
                    Action,QTable = DQN.ContextAction(State,actionnum = len(i.Neighbor))
                    if len(QTable):
                        self.TotallyMaxQ += QTable[Action]
                        self.TotallyMaxQNumber += 1

                #建立Agent实例
                NowAgent = Agent(
                                FromCluster = i,
                                ArriveCluster = None,
                                Vehicle = j,
                                State = State,
                                StateQTable = QTable,
                                Action = Action,
                                TotallyReward = None,
                                PositiveReward = None,
                                NegativeReward = None,
                                NeighborNegativeReward = None,
                                State_ = [],
                                State_QTable = None
                                )

                self.AgentTempPool.append(NowAgent)

                #调度操作
                #随机抽到达点
                #------------------
                if Action > len(i.Neighbor):
                    NowAgent.NegativeReward = -self.RewardValue
                    NowAgent.PositiveReward = -self.RewardValue
                    NowAgent.NeighborNegativeReward = -self.RewardValue
                    #非法调度，停留原地
                    NowAgent.ArriveCluster = NowAgent.FromCluster

                    '''
                    #当Action是最后一个时，指向停留在原Cluster
                    elif Action == len(i.Neighbor):
                        #explog.LogDebug("停留在原地")
                        NowAgent.ArriveCluster = NowAgent.FromCluster

                    #Action < len(i.Neighbor):
                    else:
                        ArriveCluster = i.Neighbor[Action]
                        NowAgent.ArriveCluster = ArriveCluster
                    '''

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

                        '''
                        while not len(TempCostList):
                            loopnum = 0
                            for k in range(len(ArriveCluster.Nodes)):
                                DeliveryPoint = ArriveCluster.Nodes[k][0]
                                if self.RoadCost(j.LocationNode,DeliveryPoint) < RebalanceTimeLim + np.timedelta64(loopnum * MINUTES):
                                    TempCostList.append(DeliveryPoint)
                        '''

                        while not len(TempCostList):
                            for k in range(len(ArriveCluster.Nodes)):
                                DeliveryPoint = ArriveCluster.Nodes[k][0]
                                if self.RoadCost(j.LocationNode,DeliveryPoint) < RebalanceTimeLim:
                                    TempCostList.append(DeliveryPoint)

                        DeliveryPoint = random.choice(TempCostList)

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

                        #print("j.DeliveryPoint:",j.DeliveryPoint,"cost:",self.RoadCost(j.LocationNode,j.DeliveryPoint))

                    #在所有到达点里随机选择
                    elif True:
                        j.DeliveryPoint = random.choice(ArriveCluster.Nodes)[0]

                    #PrintVehicleTrajectory(j)

                    #调度惩罚
                    #------------------------------
                    if NowAgent.NegativeReward == None:
                        NowAgent.NegativeReward = -self.RoadCost(j.LocationNode,j.DeliveryPoint)*0.2
                    else:
                        NowAgent.NegativeReward += -self.RoadCost(j.LocationNode,j.DeliveryPoint)*0.2
                    #------------------------------

                    self.TotallyRebalanceCost += self.RoadCost(j.LocationNode,j.DeliveryPoint)

                    #Delivery Cluster {Vehicle:ArriveTime}
                    ArriveCluster.VehiclesArrivetime[j] = self.RealExpTime + np.timedelta64(self.RoadCost(j.LocationNode,j.DeliveryPoint)*MINUTES)

                    #Add NeighborArriveList Information
                    #i = NowCluster  j = Rebalance Vehicles
                    #ArriveCluster.NeighborArriveList[i][j] = self.RealExpTime + np.timedelta64(self.RoadCost(j.LocationNode,j.DeliveryPoint)*MINUTES)

                    #delete now Cluster's recode about now Vehicle
                    i.IdleVehicles.remove(j)   

                    self.RebalanceNum += 1
        #------------------------------------------------

        return



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
            StepOrdersNum = len(i.Orders)
            StepOrdersSumValue = 0

            #统计每轮每个Cluster丢失的订单数量
            StepRejectNum = 0
            StepRejectSumValue = 0


            for j in i.Orders:
                if j.ArriveInfo == "Reject":
                    StepRejectNum += 1
                    StepRejectSumValue += j.OrderValue
                    #self.PrintOrder(j)
                StepOrdersSumValue += j.OrderValue

            #清除上一次Cluster内所有出现AllClusters的订单的记录
            i.Orders.clear()


            #PositiveReward
            #------------------------
            AllClusterPositiveReward = None
            #if i.PerRebalanceIdleVehicles - i.PerMatchIdleVehicles == 0:
            if i.PerMatchIdleVehicles == 0:
                AllClusterPositiveReward = 0
            else:
                #AllClusterPositiveReward = (PositiveOrdersSumValueHyperparameter * StepOrdersSumValue + PositiveOrdersNumHyperparameter * (StepOrdersNum * self.RewardValue)) / i.PerMatchIdleVehicles
                AllClusterPositiveReward = StepOrdersSumValue / i.PerMatchIdleVehicles
            #------------------------
            
            #NegativeReward
            #------------------------
            AllClusterNegativeReward = None
            if StepRejectNum > 0:
                if (i.PerRebalanceIdleVehicles - i.PerMatchIdleVehicles) == 0:
                    #AllClusterNegativeReward = -(NegativeRejectSumValueHyperparameter * StepRejectSumValue + NegativeRejectNumHyperparameter * (StepRejectNum * self.RewardValue)) / 10
                    AllClusterNegativeReward = -(StepRejectSumValue/10)
                else:
                    #AllClusterNegativeReward = -(NegativeRejectSumValueHyperparameter * StepRejectSumValue + NegativeRejectNumHyperparameter * (StepRejectNum * self.RewardValue)) / (i.PerRebalanceIdleVehicles - i.PerMatchIdleVehicles)
                    AllClusterNegativeReward = -(StepRejectSumValue/(i.PerRebalanceIdleVehicles - i.PerMatchIdleVehicles))
            else:
                AllClusterNegativeReward = 0
            #------------------------
            
            #NeighborNegativeReward
            AllClusterNeighborNegativeReward = None
            #------------------------
            if StepRejectNum > 0:
                NeighborNegativeNum = 0
                for j in i.Neighbor:
                    #包括了一些调度过来的车，稍微粗糙了点
                    #NeighborNegativeNum += (j.PerRebalanceIdleVehicles - j.PerMatchIdleVehicles)
                    NeighborNegativeNum += j.PerMatchIdleVehicles

                #AllClusterNeighborNegativeReward = NegativeRejectSumValueHyperparameter * StepRejectSumValue + NegativeRejectNumHyperparameter * (StepRejectNum * self.RewardValue)
                AllClusterNeighborNegativeReward = StepRejectSumValue

                if NeighborNegativeNum != 0:
                    AllClusterNeighborNegativeReward = -(AllClusterNeighborNegativeReward / NeighborNegativeNum)
                else :
                    AllClusterNeighborNegativeReward = -(AllClusterNeighborNegativeReward / 10)
            #------------------------
            

            #给Agent发放奖励
            for j in self.AgentTempPool:
                #对指定调度节点为当前簇，且已经到了的调度，且奖励为空才进行奖励，没到的不奖励。

                #正面奖励
                if AllClusterPositiveReward != None:
                    if j.ArriveCluster == i and j.Vehicle.Cluster == i:
                        if j.PositiveReward != None:
                            if j.PositiveReward > 0:
                                j.PositiveReward += PositiveRewardHyperparameter * AllClusterPositiveReward
                        else:
                            j.PositiveReward = PositiveRewardHyperparameter * AllClusterPositiveReward

                
                #对离开的车和未调度过来的车进行负面惩罚
                if AllClusterNegativeReward != None:
                    if j.FromCluster == i and j.Vehicle.Cluster != i:
                        if j.NegativeReward != None:
                            j.NegativeReward += NegativeRewardHyperparameter * AllClusterNegativeReward
                        else:
                            j.NegativeReward = NegativeRewardHyperparameter * AllClusterNegativeReward

                
                if AllClusterNeighborNegativeReward != None:
                    #惩罚邻居
                    if j.FromCluster in i.Neighbor and j.Vehicle.Cluster != i:
                        if j.NeighborNegativeReward != None:
                            j.NeighborNegativeReward += NegativeRewardHyperparameter * AllClusterNeighborNegativeReward
                        else:
                            j.NeighborNegativeReward = NegativeRewardHyperparameter * AllClusterNeighborNegativeReward
                
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

            if j.NegativeReward != None:
                j.TotallyReward += j.NegativeReward
            if j.NeighborNegativeReward != None:
                j.TotallyReward += j.NeighborNegativeReward
            
            self.StepReward += j.TotallyReward

            #print(j.TotallyReward,j.PositiveReward,j.NegativeReward,j.NeighborNegativeReward)
            #j.Example()

            #得到新的Cluster的State情况
            #ClusterState_ = j.ArriveCluster.GetClusterState(ClusterStateSize,self.RealExpTime)

            #得到新的Cluster的State情况
            #------------------------
            j.State_ = self.GetStateFunction(NowCluster=j.FromCluster)

            j.State_ = np.reshape(j.State_, [1, DQN.state_size])
            #------------------------
            '''
            print(j.State)
            print(j.Action)
            print(j.StateQTable)
            print(j.TotallyReward)
            print(j.State_)
            print("-----------")
            '''
            DQN.remember(j.State, j.Action, j.TotallyReward, j.State_, False)

            if SaveMemorySignal == True :
                self.SaveMemory.append((j.State, j.Action, j.TotallyReward, j.State_))


        for i in range(len(self.AgentTempPool)-1, -1, -1):
            #删除缓存池里已经完整的经验，对还没有完成旅程的经验不处理
            #if self.AgentTempPool[i].State_ != None:
            if len(self.AgentTempPool[i].State_):
            #if self.AgentTempPool[i].State_ != None and self.AgentTempPool[i].Reward != None:
                self.AgentTempPool.pop(i)

        #记录每一步更新的结束时间
        #print("更新agent池的时间:",dt.datetime.now() - StepUpdateStartTime)

        return


    def LearningFunction(self):
        #learning
        #------------------------------------------------
        #if len(DQN.memory) > DQN.batch_size:
        #if (len(DQN.memory) > DQN.batch_size) and (self.step % 12 == 0):
        if True:

            #for i in range(4096//DQN.batch_size):
            #for i in range(len(DQN.memory)//DQN.batch_size):
            for i in range(5):
                ExpLossHistory = DQN.replay(DQN.batch_size)

            if DQN.epsilon > DQN.epsilon_min:
                DQN.epsilon *= DQN.epsilon_decay

            print("loss均值: ",round(np.mean(ExpLossHistory),5),"loss方差: ",round(np.var(ExpLossHistory),5),"epsilon: ",round(DQN.epsilon,5))
            print("Average Max Q value:",round(self.TotallyMaxQ/self.TotallyMaxQNumber,3))
        #------------------------------------------------ 

        #每x次step更换参数
        #---------------------------------------------
        #if (self.step > 30) and (self.step % 15 == 0):
        if (self.Episode % 1 == 0):
            DQN.update_target_model()
        #---------------------------------------------

        return


if __name__ == "__main__":

    explog = Logger()

    #ClusterMode = "KmeansClustering"
    #ClusterMode = "SpectralClustering"
    ClusterMode = "Grid"
    DeamdPredictionMode = "CNN"
    RebalanceMode = "SmartDQN"

    #LocalRegionBound = (104.045,104.095,30.635,30.685)
    LocalRegionBound = (104.035,104.105,30.625,30.695)

    VehiclesNumber = 7000
    #500m
    SideLengthMeter = 800
    #1500m
    VehiclesServiceMeter =2000

    EXPSIM = SmartSimulation(
                        explog = explog,
                        p2pConnectedThreshold = p2pConnectedThreshold,
                        ClusterMode = ClusterMode,
                        DeamdPredictionMode = DeamdPredictionMode,
                        RebalanceMode = RebalanceMode,
                        VehiclesNumber = VehiclesNumber,
                        TimePeriods = TIMESTEP,
                        LocalRegionBound = LocalRegionBound,
                        SideLengthMeter = SideLengthMeter,
                        VehiclesServiceMeter = VehiclesServiceMeter,
                        TimeAndWeatherOneHotSignal = TimeAndWeatherOneHotSignal,
                        NeighborCanServer = NeighborCanServer,
                        FocusOnLocalRegion = FocusOnLocalRegion,
                        SaveMemorySignal = SaveMemorySignal
                        )

    EXPSIM.CreateAllInstantiate()

    action_size = 0
    for i in EXPSIM.Clusters:
        if len(i.Neighbor) > action_size :
            #include himself
            action_size = len(i.Neighbor)

    #include himself
    action_size += 1

    #5(时间和天气特征) ClusterIDBinarySize + 2(ID,自身空闲车辆,下个时间点将要到达的出租车)  (action_size - 1) * 3(邻居目前的空闲，邻居到我的调度数目，下个X时间内将到邻居的车数)

    #大状态，邻居计算限定时间到达
    #state_size = 5 + ClusterIDBinarySize + 2 + (action_size) * 3
    #ClusterStateSize = ClusterIDBinarySize + 2 + (action_size) * 3

    state_size = ClusterIDBinarySize + action_size * 2

    print("state_size",state_size,"action_size",action_size)

    DQN = CNNDQNAgent(
                    state_size = state_size,
                    action_size = action_size,
                    #memory_size = deque(maxlen=4096),
                    memory_size = deque(maxlen=8192),
                    #memory_size = deque(maxlen=1024),
                    gamma = 0.95,
                    epsilon = 0.40,
                    epsilon_min = 0.01,
                    #epsilon_decay = 0.999,
                    epsilon_decay = 0.999,
                    learning_rate = 0.0005,
                    #batch_size = 256
                    batch_size = 32
                    )

    #DQN.load("./model/"+str(EXPSIM.ClusterMode)+str(EXPSIM.DeamdPredictionMode)+str(EXPSIM.SideLengthMeter)+str(EXPSIM.LocalRegionBound)+".h5")

    OrderFileDate = ["1101"]

    while EXPSIM.Episode < 500:
        if EXPSIM.Episode == 100:
            DQN.learning_rate = 0.0001

        if SaveMemorySignal == True :
            EXPSIM.ReadSaveNum(filepath = "./remember/new")
        EXPSIM.SimCity(SaveMemorySignal)

        print("DQN.epsilon",DQN.epsilon)

        DQN.save("./model/"+str(EXPSIM.ClusterMode)+str(EXPSIM.DeamdPredictionMode)+
                            str(EXPSIM.SideLengthMeter)+str(EXPSIM.LocalRegionBound)+".h5")

        if SaveMemorySignal == True :
            EXPSIM.SaveNumber= EXPSIM.SaveNumber + 1

        EXPSIM.Reset(OrderFileDate[0])

        EXPSIM.Episode += 1



