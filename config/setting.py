import numpy as np

#Simulater Setting
#------------------------------
MINUTES=60000000000
TIMESTEP = np.timedelta64(10*MINUTES)
PICKUPTIMEWINDOW = np.timedelta64(10*MINUTES)

#It can enable the neighbor car search system to determine the search range according to the set search distance and the size of the grid.
#It use dfs to find the nearest idle vehicles in the area.
NeighborCanServer = False

#You can adjust the size of the experimental area by entering latitude and longitude.
#The order, road network and grid division will be adaptive. Adjust to fit selected area
FocusOnLocalRegion = False
LocalRegionBound = (104.035,104.105,30.625,30.695)
if FocusOnLocalRegion == False:
    LocalRegionBound = (104.011, 104.125, 30.618, 30.703)



#Input parameters
VehiclesNumber = 6000
SideLengthMeter = 800
VehiclesServiceMeter = 800

DispatchMode = "Simulation"
DemandPredictionMode = "None"
#["TransportationClustering","KmeansClustering","SpectralClustering"]
ClusterMode = "Grid"

