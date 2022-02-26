from util import DataModule
from config.setting import (
    TIMESTEP,
    NELGHBOR_CAN_SERVER,
    LOCAL_REGION_BOUND,
    VEHICLES_NUMBER,
    SIDE_LENGTH_KIRO_METER,
    VEHICLE_SERVICE_KIRO_METER,
    DISPATCH_MODE,
    DEMAND_PREDICTION_MODE,
    AREA_MODE,
)
from simulator.simulator import Simulation

if __name__ == "__main__":
    simulator = Simulation(
        area_mode=AREA_MODE,
        demand_prediction_mode=DEMAND_PREDICTION_MODE,
        dispatch_mode=DISPATCH_MODE,
        vehicles_number=VEHICLES_NUMBER,
        time_periods=TIMESTEP,
        local_region_bound=LOCAL_REGION_BOUND,
        side_length_meter=SIDE_LENGTH_KIRO_METER,
        vehicles_server_meter=VEHICLE_SERVICE_KIRO_METER,
        neighbor_can_server=NELGHBOR_CAN_SERVER,
    )

    date_module = DataModule()
    simulator.create_all_instantiate(date_module.date)
    simulator()

    while date_module.next():
        simulator.reload(date_module.date)
        simulator()
