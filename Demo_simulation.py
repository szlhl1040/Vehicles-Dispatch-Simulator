
from config.setting import *
from simulator.simulator import Simulation

if __name__ == "__main__":
    EXPSIM = Simulation(
        cluster_mode = ClusterMode,
        demand_prediction_mode = DemandPredictionMode,
        dispatch_mode = DispatchMode,
        vehicles_number = VehiclesNumber,
        TimePeriods = TIMESTEP,
        LocalRegionBound = LocalRegionBound,
        SideLengthMeter = SideLengthMeter,
        VehiclesServiceMeter = VehiclesServiceMeter,
        NeighborCanServer = NeighborCanServer,
        FocusOnLocalRegion = FocusOnLocalRegion,
    )

    EXPSIM.CreateAllInstantiate()
    EXPSIM.SimCity()
    EXPSIM.Reload("0602")
    EXPSIM.SimCity()
