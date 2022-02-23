import numpy as np
from domain.area_mode import AreaMode
from domain.demand_prediction_mode import DemandPredictionMode
from domain.local_region_bound import LocalRegionBound

#Simulater Setting
#------------------------------
MINUTES=60000000000
TIMESTEP = np.timedelta64(10*MINUTES)
PICKUPTIMEWINDOW = np.timedelta64(10*MINUTES)

#It can enable the neighbor car search system to determine the search range according to the set search distance and the size of the grid.
#It use dfs to find the nearest idle vehicles in the area.
NELGHBOR_CAN_SERVER = False

#You can adjust the size of the experimental area by entering latitude and longitude.
#The order, road network and grid division will be adaptive. Adjust to fit selected area
FOCUS_ON_LOCAL_REGION = True
LOCAL_REGION_BOUND = LocalRegionBound(
    west_bound=-74.020,
    east_bound=-73.950,
    south_bound=40.700,
    north_bound=40.770
)
if FOCUS_ON_LOCAL_REGION:
    LOCAL_REGION_BOUND = LocalRegionBound(
        west_bound=-74.020,
        east_bound=-74.010,
        south_bound=40.70,
        north_bound=40.710
    )



#Input parameters
VEHICLES_NUMBER = 6000
SIDE_LENGTH_METER = 800
VEHICLE_SERVICE_METER = 800

DISPATCH_MODE = "Simulation"
DEMAND_PREDICTION_MODE = DemandPredictionMode.TRAINING
#["TransportationClustering","KmeansClustering","SpectralClustering"]
AREA_MODE = AreaMode.GRID
