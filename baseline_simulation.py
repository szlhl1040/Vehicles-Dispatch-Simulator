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
from config.setting import *
from simulator.simulator import Logger,Simulation


###########################################################################

if __name__ == "__main__":

    explog = Logger()
    p2pConnectedThreshold = 0.8
    RebalanceMode = "Simulation"
    DeamdPredictionMode = "None"
    
    VehiclesNumber = 7000

    #500m
    SideLengthMeter = 800
    #1500m
    VehiclesServiceMeter = 2000

    TIMESTEP = np.timedelta64(10*MINUTES)

    #LocalRegionBound = (104.045,104.095,30.635,30.685)
    LocalRegionBound = (104.035,104.105,30.625,30.695)


    ClusterMode = "KmeansClustering"
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
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM


    ClusterMode = "TransportationClustering"
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
    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM


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

    EXPSIM.CreateAllInstantiate()
    #EXPSIM.CalculateInterTransportation()
    EXPSIM.SimCity()
    del EXPSIM