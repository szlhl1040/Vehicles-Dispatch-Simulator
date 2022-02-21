
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
    simulator = Simulation(
        cluster_mode = ClusterMode,
        demand_prediction_mode = DemandPredictionMode,
        dispatch_mode = DispatchMode,
        vehicles_number = VehiclesNumber,
        time_periods = TIMESTEP,
        local_region_bound = LocalRegionBound,
        side_length_meter = SideLengthMeter,
        vehicles_server_meter = VehiclesServiceMeter,
        neighbor_can_server = NeighborCanServer,
        focus_on_local_region = FocusOnLocalRegion,
    )

    simulator.create_all_instantiate()
    simulator.simulate()
    simulator.reload("0602")
    simulator.simulate()
