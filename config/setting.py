import os
import numpy as np


#Simulater Setting
#------------------------------------------

#Biggest Cluster ID is 255
ClusterIDBinarySize = 8
#------------------------------------------


#Reawrd Setting
#------------------------------------------
PositiveOrdersSumValueHyperparameter = 0.8

PositiveOrdersNumHyperparameter = 0.2

NegativeRejectSumValueHyperparameter = 0.8

NegativeRejectNumHyperparameter = 0.2

PositiveRewardHyperparameter = 0.6

NegativeRewardHyperparameter = 0.4
#------------------------------------------


MINUTES=60000000000

EARTH_REDIUS = 6378.137

PI = 3.141592653589793

#东西方向上一米等于的经度
#360 / 40075016.68557849
#Meter2Longitude = 0.000009405717451407729
Meter2Longitude = 0.00001141

#南北方向上一米等于的纬度
#360 / 38274592.22115159
#Meter2Latitude = 0.000008983152841195214
Meter2Latitude = 0.00000899

TIMESTEPNUMBER = 10

TransportLimtedTimeConnectedThreshold = 10 #10min

TIMESTEP = np.timedelta64(10*MINUTES)
#TIMESTEP = np.timedelta64(30*MINUTES)

RebalanceTimeLim = np.timedelta64(10*MINUTES)

PICKUPTIMEWINDOW = np.timedelta64(10*MINUTES)

p2pConnectedThreshold = 0.8

PATH = "./"


#Agent Setting
#------------------------------------------
#state_size = 3

#action_size = 0
#------------------------------------------

PrintAllClusterSignal = False

SaveClusterSignal = False

#ClusterMode = "Cluster"

SaveMemorySignal = False

NeighborCanServer = False

FocusOnLocalRegion = True

TimeAndWeatherOneHotSignal = True


