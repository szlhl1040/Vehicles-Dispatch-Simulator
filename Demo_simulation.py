
from config.setting import *
from simulator.simulator import Simulation

if __name__ == "__main__":
    EXPSIM = Simulation(
        ClusterMode = ClusterMode,
        DemandPredictionMode = DemandPredictionMode,
        DispatchMode = DispatchMode,
        VehiclesNumber = VehiclesNumber,
        TimePeriods = TIMESTEP,
        LocalRegionBound = LocalRegionBound,
        SideLengthMeter = SideLengthMeter,
        VehiclesServiceMeter = VehiclesServiceMeter,
        NeighborCanServer = NeighborCanServer,
        FocusOnLocalRegion = FocusOnLocalRegion,
    )

    EXPSIM.CreateAllInstantiate()
    EXPSIM.SimCity()

