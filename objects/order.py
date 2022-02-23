class Order(object):
    
    def __init__(
        self,
        id,
        ReleasTime,
        pickup_point,
        delivery_point,
        PickupTimeWindow,
        PickupWaitTime,
        ArriveInfo,
        order_value
    ):
        self.id = id                                            #This order's ID 
        self.ReleasTime = ReleasTime                            #Start time of this order
        self.pickup_point = pickup_point                          #The starting position of this order
        self.delivery_point = delivery_point                      #Destination of this order
        self.PickupTimeWindow = PickupTimeWindow                #Limit of waiting time for this order
        self.PickupWaitTime = PickupWaitTime                    #This order's real waiting time from running in the simulator
        self.ArriveInfo = ArriveInfo                            #Processing information for this order
        self.order_value = order_value                            #The value of this order

    def arrive_order_time_record(self, ArriveTime):
        self.ArriveInfo = "ArriveTime:"+str(ArriveTime)

    def example(self):
        print("Order Example output")
        print("ID:",self.id)
        print("ReleasTime:",self.ReleasTime)
        print("PickupPoint:",self.pickup_point)
        print("DeliveryPoint:",self.delivery_point)
        print("PickupTimeWindow:",self.PickupTimeWindow)
        print("PickupWaitTime:",self.PickupWaitTime)
        print("ArriveInfo:",self.ArriveInfo)
        print()

    def reset(self):
        self.PickupWaitTime = None
        self.ArriveInfo = None