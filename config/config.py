from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from domain import AreaMode, DemandPredictionMode, DispatchMode, LocalRegionBound


@dataclass(frozen=True)
class Config:
    MINUTES: int
    NELGHBOR_CAN_SERVER: bool
    LOCAL_REGION_BOUND: LocalRegionBound
    VEHICLES_NUMBER: int
    SIDE_LENGTH_KIRO_METER: float
    VEHICLE_SERVICE_KIRO_METER: float
    DISPATCH_MODE: DispatchMode
    DEMAND_PREDICTION_MODE: DemandPredictionMode
    AREA_MODE: AreaMode

    @property
    def TIMESTEP(self):
        return np.timedelta64(10 * self.MINUTES)

    @property
    def PICKUPTIMEWINDOW(self):
        return np.timedelta64(10 * self.MINUTES)

    @staticmethod
    def load() -> "Config":
        with open(Path(__file__).parent / "config.yaml") as f:
            config = yaml.safe_load(f)
        use_bounds = config["LOCAL_REGION_BOUND"][config["DATA_SIZE"]]
        LOCAL_REGION_BOUND = LocalRegionBound(
            west_bound=use_bounds["west"],
            east_bound=use_bounds["east"],
            south_bound=use_bounds["south"],
            north_bound=use_bounds["north"],
        )
        return Config(
            MINUTES=config["MINUTES"],
            NELGHBOR_CAN_SERVER=config["NELGHBOR_CAN_SERVER"],
            LOCAL_REGION_BOUND=LOCAL_REGION_BOUND,
            VEHICLES_NUMBER=config["VEHICLES_NUMBER"],
            SIDE_LENGTH_KIRO_METER=config["SIDE_LENGTH_KIRO_METER"],
            VEHICLE_SERVICE_KIRO_METER=config["VEHICLE_SERVICE_KIRO_METER"],
            DISPATCH_MODE=DispatchMode(config["DISPATCH_MODE"]),
            DEMAND_PREDICTION_MODE=DemandPredictionMode(
                config["DEMAND_PREDICTION_MODE"]
            ),
            AREA_MODE=AreaMode(config["AREA_MODE"]),
        )
