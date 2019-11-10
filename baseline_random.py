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
from objects.objects import Cluster,Order,Vehicle,Agent,Grid
from simulator.simulator import Logger,Simulation
from config.setting import *



###########################################################################


class Simulation(Simulation):

    def DefiningLocalVariables(self,RebalanceProbability):
        self.RebalanceProbability = RebalanceProbability

        return

    #---------------------------------------------------------------------------
    def RebalanceFunction(self):
        #Policy
        #------------------------------------------------
        TempAllClusters = self.Clusters[:]
        random.shuffle(TempAllClusters)

        #i = each Cluster
        for i in TempAllClusters:

            if self.RealExpTime > self.Orders[-1].ReleasTime:
                break

            if not len(i.Neighbor):
                continue

            for j in i.IdleVehicles:
                #按概率随机调度
                if random.choice(range(1,101)) <= self.RebalanceProbability:
                    Action = random.choice(i.Neighbor)
                else:
                    Action = i

                #当Action不是0 和 Action内有路点时，指向停留在原Cluster
                if Action != i and len(Action.Nodes):
                    ArriveCluster = Action

                    #从限定时间内的到达点（N * 10min）里随机选择
                    if False:
                        TempCostList = []
                        while not len(TempCostList):
                            loopnum = 0
                            for k in range(len(ArriveCluster.Nodes)):
                                DeliveryPoint = ArriveCluster.Nodes[k][0]
                                if self.RoadCost(j.LocationNode,DeliveryPoint) < RebalanceTimeLim + np.timedelta64(loopnum * MINUTES):
                                    TempCostList.append(DeliveryPoint)

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

                        if len(TempCostList) <= 5:
                            j.DeliveryPoint = random.choice(TempCostList)
                        elif len(TempCostList) > 5 and len(TempCostList) <= 10:
                            j.DeliveryPoint = random.choice(TempCostList[5:10])
                        elif len(TempCostList) > 10:
                            j.DeliveryPoint = random.choice(TempCostList[:10])
                            

                    #在所有到达点里随机选择
                    elif True:
                        j.DeliveryPoint = random.choice(ArriveCluster.Nodes)[0]


                    if j.LocationNode != None and j.DeliveryPoint != None :
                        self.TotallyRebalanceCost += self.RoadCost(j.LocationNode,j.DeliveryPoint)

                        #Delivery Cluster {Vehicle:ArriveTime}
                        ArriveCluster.VehiclesArrivetime[j] = self.RealExpTime + np.timedelta64(self.RoadCost(j.LocationNode,j.DeliveryPoint)*MINUTES)

                        #delete now Cluster's recode about now Vehicle
                        i.IdleVehicles.remove(j)   

                        self.RebalanceNum += 1
        #------------------------------------------------
        return



if __name__ == "__main__":

    explog = Logger()
    p2pConnectedThreshold = 0.8
    RebalanceMode = "Random"
    #  x% Probability to rebalance
    RebalanceProbability = 50
    DeamdPredictionMode = "None"

    VehiclesNumber = 7000

    #500m
    SideLengthMeter = 800
    #1500m
    VehiclesServiceMeter =2000

    TIMESTEP = np.timedelta64(10*MINUTES)

    #LocalRegionBound = (104.045,104.095,30.635,30.685)
    LocalRegionBound = (104.035,104.105,30.625,30.695)

    ClusterMode = "KmeansClustering"
    ClusterMode = "TransportationClustering2"
    EXPSIM = Simulation(
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
    EXPSIM.DefiningLocalVariables(RebalanceProbability)
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM


    ClusterMode = "TransportationClustering"
    ClusterMode = "Grid"
    EXPSIM = Simulation(
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
    EXPSIM.DefiningLocalVariables(RebalanceProbability)
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM

    exit()


    ClusterMode = "TransportationClustering2"
    EXPSIM = Simulation(
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
    EXPSIM.DefiningLocalVariables(RebalanceProbability)
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM


    ClusterMode = "SpectralClustering"
    EXPSIM = Simulation(
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
    EXPSIM.DefiningLocalVariables(RebalanceProbability)
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM


    ClusterMode = "Grid"
    EXPSIM = Simulation(
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
    EXPSIM.DefiningLocalVariables(RebalanceProbability)
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM


