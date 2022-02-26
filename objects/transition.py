class Transition(object):
    def __init__(
        self,
        FromCluster,
        ArriveCluster,
        Vehicle,
        State,
        StateQTable,
        Action,
        TotallyReward,
        PositiveReward,
        NegativeReward,
        NeighborNegativeReward,
        State_,
        State_QTable,
    ):
        self.FromCluster = FromCluster
        self.ArriveCluster = ArriveCluster
        self.Vehicle = Vehicle
        self.State = State
        self.StateQTable = StateQTable
        self.Action = Action
        self.TotallyReward = TotallyReward
        self.PositiveReward = PositiveReward
        self.NegativeReward = NegativeReward
        self.NeighborNegativeReward = NeighborNegativeReward
        self.State_ = State_
        self.State_QTable = State_QTable

    def example(self):
        print("Transition Example output")
        print("Action:", self.Action)
        print("TotallyReward:", self.TotallyReward)
        print("PositiveReward:", self.PositiveReward)
        print("NegativeReward:", self.NegativeReward)
        print("NeighborNegativeReward:", self.NeighborNegativeReward)
        print()
