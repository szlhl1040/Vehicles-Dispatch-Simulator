import numpy as np
import pandas as pd
import random
from config.setting import *

class Cluster(object):

    def __init__(self,ID,Nodes,Neighbor,RebalanceNumber,IdleVehicles,VehiclesArrivetime,Orders):
        self.ID = ID
        self.Nodes = Nodes
        self.Neighbor = Neighbor
        self.RebalanceNumber = RebalanceNumber
        self.IdleVehicles = IdleVehicles
        self.VehiclesArrivetime = VehiclesArrivetime
        self.Orders = Orders
        self.PerRebalanceIdleVehicles = 0
        self.LaterRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0
        self.RebalanceFrequency = 0

    def Reset(self):
        self.RebalanceNumber = 0
        self.IdleVehicles.clear()
        self.VehiclesArrivetime.clear()
        self.Orders.clear()
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0

    def ArriveClusterUpDate(self, vehicle):
        self.IdleVehicles.append(vehicle)
        self.VehiclesArrivetime.pop(vehicle)

    def Example(self):
        print("Order Example output")
        print("ID:",self.ID)
        print("Nodes:",self.Nodes)
        print("Neighbor:",self.Neighbor)
        print("RebalanceNumber:",self.RebalanceNumber)
        print("IdleVehicles:",self.IdleVehicles)
        print("VehiclesArrivetime:",self.VehiclesArrivetime)
        print("Orders:",self.Orders)
        

class Order(object):

    def __init__(self,ID,ReleasTime,PickupPoint,DeliveryPoint,PickupTimeWindow,PickupWaitTime,ArriveInfo,OrderValue):
        self.ID = ID                                            #This order's ID 
        self.ReleasTime = ReleasTime                            #Start time of this order
        self.PickupPoint = PickupPoint                          #The starting position of this order
        self.DeliveryPoint = DeliveryPoint                      #Destination of this order
        self.PickupTimeWindow = PickupTimeWindow                #Limit of waiting time for this order
        self.PickupWaitTime = PickupWaitTime                    #This order's real waiting time from running in the simulator
        self.ArriveInfo = ArriveInfo                            #Processing information for this order
        self.OrderValue = OrderValue                            #The value of this order

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

    def Reset(self):
        self.PickupWaitTime = None
        self.ArriveInfo = None


class Vehicle(object):

    def __init__(self,ID,LocationNode,Cluster,Orders,DeliveryPoint):
        self.ID = ID                                            #This vehicle's ID    
        self.LocationNode = LocationNode                        #Current vehicle's location
        self.Cluster = Cluster                                  #Which cluster the current vehicle belongs to
        self.Orders = Orders                                    #Orders currently on board
        self.DeliveryPoint = DeliveryPoint                      #Next destination of current vehicle

    def ArriveVehicleUpDate(self, DeliveryCluster):
        self.LocationNode = self.DeliveryPoint
        self.DeliveryPoint = None
        self.Cluster = DeliveryCluster
        if len(self.Orders):
            self.Orders.clear()

    def Reset(self):
        self.Orders.clear()
        self.DeliveryPoint = None

    def Example(self):
        print("Vehicle Example output")
        print("ID:",self.ID)
        print("LocationNode:",self.LocationNode)
        print("Cluster:",self.Cluster)
        print("Orders:",self.Orders)
        print("DeliveryPoint:",self.DeliveryPoint)
        print()


class Transition(object):

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
        print("Transition Example output")
        print("Action:",self.Action)
        print("TotallyReward:",self.TotallyReward)
        print("PositiveReward:",self.PositiveReward)
        print("NegativeReward:",self.NegativeReward)
        print("NeighborNegativeReward:",self.NeighborNegativeReward)
        print()


class Grid(object):

    def __init__(self,ID,Nodes,Neighbor,RebalanceNumber,IdleVehicles,VehiclesArrivetime,Orders):
        self.ID = ID
        self.Nodes = Nodes
        self.Neighbor = Neighbor
        self.RebalanceNumber = RebalanceNumber
        self.IdleVehicles = IdleVehicles
        self.VehiclesArrivetime = VehiclesArrivetime
        self.Orders = Orders
        self.PerRebalanceIdleVehicles = 0
        self.LaterRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0

    def Reset(self):
        self.RebalanceNumber = 0
        self.IdleVehicles.clear()
        self.VehiclesArrivetime.clear()
        self.Orders.clear()
        self.PerRebalanceIdleVehicles = 0
        self.PerMatchIdleVehicles = 0

    def ArriveClusterUpDate(self, vehicle):
        self.IdleVehicles.append(vehicle)
        self.VehiclesArrivetime.pop(vehicle)
        
    def Example(self):
        print("ID:",self.ID)
        print("Nodes:",self.Nodes)
        print("Neighbor:[",end=' ')
        for i in self.Neighbor:
            print(i.ID,end=' ')
        print("]")
        print("RebalanceNumber:",self.RebalanceNumber)
        print("IdleVehicles:",self.IdleVehicles)
        print("VehiclesArrivetime:",self.VehiclesArrivetime)
        print("Orders:",self.Orders)
        print()

