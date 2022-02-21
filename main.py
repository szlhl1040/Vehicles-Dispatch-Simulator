from util import DataModule
from config.setting import (
    TIMESTEP,
    NeighborCanServer,
    FocusOnLocalRegion,
    LocalRegionBound,
    VehiclesNumber,
    SideLengthMeter,
    VehiclesServiceMeter,
    DispatchMode,
    DemandPredictionMode,
    ClusterMode,
)
from simulator.simulator import Simulation

if __name__ == "__main__":
    EXPSIM = Simulation(
        cluster_mode = ClusterMode,
        demand_prediction_mode = DemandPredictionMode,
        dispatch_mode = DispatchMode,
        vehicles_number = VehiclesNumber,
        time_periods = TIMESTEP,
        LocalRegionBound = LocalRegionBound,
        SideLengthMeter = SideLengthMeter,
        VehiclesServiceMeter = VehiclesServiceMeter,
        NeighborCanServer = NeighborCanServer,
        FocusOnLocalRegion = FocusOnLocalRegion,
    )

    date_module = DataModule()
    EXPSIM.CreateAllInstantiate(date_module.date)

    while date_module.next():
        EXPSIM.Reload(date_module.date)
        EXPSIM.SimCity()
