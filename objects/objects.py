import numpy as np
import pandas as pd
import random
from config.setting import *

class Cluster(object):

    def __init__(self,ID,Nodes,Neighbor,NeighborArriveList,IdleVehicles,VehiclesArrivetime,Orders):
        self.ID = ID
        self.Nodes = Nodes
        self.Neighbor = Neighbor
        self.NeighborArriveList = NeighborArriveList	#{{},{},{}}#{v1:arrivetime,....}//{t1:[v:node],t2:[v:node]...}
        self.IdleVehicles = IdleVehicles
        self.VehiclesArrivetime = VehiclesArrivetime
        self.Orders = Orders
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0


    def Reset(self):
        self.NeighborArriveList.clear()
        self.IdleVehicles.clear()
        self.VehiclesArrivetime.clear()
        self.Orders.clear()
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0


    def ArriveClusterUpDate(self, vehicle):
        self.IdleVehicles.append(vehicle)
        self.VehiclesArrivetime.pop(vehicle)


    def Int2BinList(self, Int):
        BinStr = bin(Int).replace('0b','')

        BinList = []
        for i in BinStr:
            BinList.append(int(i))
        for i in range(64):
            if len(BinList) == ClusterIDBinarySize :
                break
            elif len(BinList) > ClusterIDBinarySize:
                raise Exception('Cluster ID Binary Size Error!')
            else :
                BinList.insert(0,0)
        return BinList


    def GetClusterState(self,ClusterStateSize,RealExpTime):

        #RebalanceTimeLim = 10min

        IntervalSize = 5

        if True:
            EffVehiclesArriveNum = 0
            for key in self.VehiclesArrivetime:
                if self.VehiclesArrivetime[key] <= RealExpTime + RebalanceTimeLim:
                    EffVehiclesArriveNum += 1


            NowClusterState = self.Int2BinList(self.ID)
            NowClusterState += [len(self.IdleVehicles)//IntervalSize,EffVehiclesArriveNum//IntervalSize,len(self.Orders)//IntervalSize]
        else:
            NowClusterState = self.Int2BinList(self.ID)
            NowClusterState += [self.ID,len(self.IdleVehicles)//IntervalSize,len(self.VehiclesArrivetime)//IntervalSize,len(self.Orders)//IntervalSize]


        if True:
            NeighborClusterState = []

            #print(len(self.Neighbor))

            for i in self.Neighbor:
                NeighborClusterState.append(len(i.IdleVehicles)//IntervalSize)


                #!!!!!!!
                '''
                print(self.NeighborArriveList[i])
                print(self.NeighborArriveList)

                temp = 0
                for key in self.NeighborArriveList :
                    temp += len(self.NeighborArriveList[key])

                print(self.NeighborArriveList)
                '''

                #NeighborClusterState.append(len(self.NeighborArriveList[i])//IntervalSize)

                NeighborClusterState.append(len(self.VehiclesArrivetime)//IntervalSize)

                #!!!!!!!

                if False:
                    EffVehiclesArriveNum = 0
                    for key in i.VehiclesArrivetime:
                        if i.VehiclesArrivetime[key] <= RealExpTime + RebalanceTimeLim:
                            EffVehiclesArriveNum += 1

                    NeighborClusterState.append(EffVehiclesArriveNum)
                    

            ClusterState = NowClusterState + NeighborClusterState

            #print(len(ClusterState),ClusterState)

            #fill ClusterState use ZERO
            while (True) :
                if len(ClusterState) < ClusterStateSize :
                    #当ClusterState不足时，用0填充
                    ClusterState.append(0)
                elif len(ClusterState) == ClusterStateSize:
                    break
                else :
                    raise Exception('Cluster State Size Error!')

            #print(len(ClusterState),ClusterState)

            return ClusterState

        else:
            return NowClusterState



    def Example(self):
        print("Order Example output")
        print("ID:",self.ID)
        print("Nodes:",self.Nodes)
        print("Neighbor:",self.Neighbor)
        print("NeighborArriveList:",self.NeighborArriveList)
        print("IdleVehicles:",self.IdleVehicles)
        print("VehiclesArrivetime:",self.VehiclesArrivetime)
        print("Orders:",self.Orders)
        



class Order(object):

    def __init__(self,ID,ReleasTime,PickupPoint,DeliveryPoint,PickupTimeWindow,PickupWaitTime,ArriveInfo,OrderValue):
        self.ID = ID                                            #请求编号
        self.ReleasTime = ReleasTime                            #请求发出时间
        self.PickupPoint = PickupPoint                          #请求开始位置
        self.DeliveryPoint = DeliveryPoint                      #请求结束位置
        self.PickupTimeWindow = PickupTimeWindow                #Q.wp {Q.wp.e:...,Q.wp.l...}
        self.PickupWaitTime = PickupWaitTime
        self.ArriveInfo = ArriveInfo
        self.OrderValue = OrderValue



    def ArriveOrderTimeRecord(self, ArriveTime):
        self.ArriveInfo = "ArriveTime:"+str(ArriveTime)


    def Example(self):
        print("Order Example output")
        print("ID:",self.ID)
        print("ReleasTime:",self.ReleasTime)
        print("PickupPoint:",self.PickupPoint)
        print("DeliveryPoint:",self.DeliveryPoint)
        print("PickupTimeWindow:",self.PickupTimeWindow)
        print("PickupWaitTime:",self.PickupWaitTime)
        print("ArriveInfo:",self.ArriveInfo)
        print()
        #exit()


class Vehicle(object):

    def __init__(self,ID,LocationNode,Cluster,Orders,DeliveryPoint):
        self.ID = ID
        self.LocationNode = LocationNode
        self.Cluster = Cluster
        self.Orders = Orders
        self.DeliveryPoint = DeliveryPoint



    def ArriveVehicleUpDate(self, DeliveryCluster):
        self.LocationNode = self.DeliveryPoint
        self.DeliveryPoint = None
        self.Cluster = DeliveryCluster

        if len(self.Orders):
            #self.Orders.pop(0)
            self.Orders.clear()
            #self.Orders.pop(self.Orders.index(self.Orders[0]))



class Agent(object):

    def __init__(self,FromCluster,ArriveCluster,Vehicle,State,StateQTable,Action,TotallyReward,PositiveReward,NegativeReward,NeighborNegativeReward,State_,State_QTable):
        self.FromCluster = FromCluster
        self.ArriveCluster = ArriveCluster
        self.Vehicle = Vehicle
        self.State = State
        self.StateQTable = StateQTable
        self.Action = Action
        self.TotallyReward = TotallyReward
        self.PositiveReward = PositiveReward
        self.NegativeReward = NegativeReward
        self.NeighborNegativeReward = NeighborNegativeReward
        self.State_ = State_
        self.State_QTable = State_QTable


    def Example(self):
        print("Action:",self.Action)
        print("TotallyReward:",self.TotallyReward)
        print("PositiveReward:",self.PositiveReward)
        print("NegativeReward:",self.NegativeReward)
        print("NeighborNegativeReward:",self.NeighborNegativeReward)
        print()


class Grid(object):

    def __init__(self,ID,Nodes,Neighbor,NeighborArriveList,IdleVehicles,VehiclesArrivetime,Orders):
        self.ID = ID
        self.Nodes = Nodes
        self.Neighbor = Neighbor
        self.NeighborArriveList = NeighborArriveList    #{v1:arrivetime,....}//{t1:[v:node],t2:[v:node]...}
        self.IdleVehicles = IdleVehicles
        self.VehiclesArrivetime = VehiclesArrivetime
        self.Orders = Orders
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0


    def Reset(self):
        self.NeighborArriveList.clear()
        self.IdleVehicles.clear()
        self.VehiclesArrivetime.clear()
        self.Orders.clear()
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0
        

    def Example(self):
        #print("Order Example output")
        print("ID:",self.ID)
        print("Nodes:",self.Nodes)
        print("Neighbor:[",end=' ')
        for i in self.Neighbor:
            print(i.ID,end=' ')
        print("]")
        print("NeighborArriveList:",self.NeighborArriveList)
        print("IdleVehicles:",self.IdleVehicles)
        print("VehiclesArrivetime:",self.VehiclesArrivetime)
        print("Orders:",self.Orders)
        print()


    def ArriveClusterUpDate(self, vehicle):
        self.IdleVehicles.append(vehicle)
        self.VehiclesArrivetime.pop(vehicle)


    def Int2BinList(self, Int):
        BinStr = bin(Int).replace('0b','')

        BinList = []
        for i in BinStr:
            BinList.append(int(i))
        for i in range(64):
            if len(BinList) == ClusterIDBinarySize :
                break
            elif len(BinList) > ClusterIDBinarySize:
                raise Exception('Cluster ID Binary Size Error!')
            else :
                BinList.insert(0,0)
        return BinList


    def GetClusterState(self,GridStateSize,RealExpTime):
        #RebalanceTimeLim = 10min
        IntervalSize = 5

        if True:
            EffVehiclesArriveNum = 0
            for key in self.VehiclesArrivetime:
                if self.VehiclesArrivetime[key] <= RealExpTime + RebalanceTimeLim:
                    EffVehiclesArriveNum += 1

            NowGridState = [self.ID,len(self.IdleVehicles)//IntervalSize,EffVehiclesArriveNum//IntervalSize]
        else:
            NowGridState = [self.ID,len(self.IdleVehicles)//IntervalSize,len(self.VehiclesArrivetime)//IntervalSize]


        if True:
            NeighborGridState = []

            #print(len(self.Neighbor))

            for i in self.Neighbor:

                #if i == None :
                #break

                NeighborGridState.append(len(i.IdleVehicles)//IntervalSize)

                #NeighborGridState.append(len(self.NeighborArriveList[i])//IntervalSize)
                #for key in i.VehiclesArrivetime

                NeighborGridState.append(len(self.NeighborArriveList)//IntervalSize)

                if False:
                    EffVehiclesArriveNum = 0
                    for key in i.VehiclesArrivetime:
                        if i.VehiclesArrivetime[key] <= RealExpTime + RebalanceTimeLim:
                            EffVehiclesArriveNum += 1

                    NeighborGridState.append(EffVehiclesArriveNum)
                    

            GridState = NowGridState + NeighborGridState

            #print(len(ClusterState),ClusterState)

            #fill ClusterState use ZERO
            while (True) :
                if len(GridState) < GridStateSize :
                    #当ClusterState不足时，用0填充
                    GridState.append(0)
                elif len(GridState) == GridStateSize:
                    break
                else :
                    raise Exception('Cluster State Size Error!')

            return GridState
        else:
            return NowGridState





if __name__ == '__main__':
    pass

