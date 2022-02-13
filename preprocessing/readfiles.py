import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime

def timestamp_datetime(value):
    d = datetime.fromtimestamp(value)
    t = dt.datetime(d.year,d.month,d.day,d.hour,d.minute,0)
    return t

def string_datetime(value):
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

def string_pdTimestamp(value):
    d = string_datetime(value)
    t = pd.Timestamp(d.year, d.month, d.day, d.hour, d.minute)
    return t

def ReadMap(input_file_path):
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Map = []
    for chunk in reader:
        Map.append(chunk)
    Map = pd.concat(Map)
    Map = Map.drop(["Unnamed: 0"], axis=1)
    Map = Map.values
    Map = Map.astype('int64')
    return Map

def ReadCostMap(input_file_path):
    reader = pd.read_csv(input_file_path,header=None,chunksize = 1000)
    Map = []
    for chunk in reader:
        Map.append(chunk)
    Map = pd.concat(Map)
    return Map

def ReadPath(input_file_path):
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Path = []
    for chunk in reader:
        Path.append(chunk)
    Path = pd.concat(Path)
    Path = Path.drop(["Unnamed: 0"], axis=1)
    Path = Path.values
    Path = Path.astype('int64')
    return Path

def ReadNode(input_file_path):
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Node = []
    for chunk in reader:
        Node.append(chunk)
    Node = pd.concat(Node)
    return Node

def ReadNodeIDList(input_file_path):
    NodeIDList = []
    with open(input_file_path, 'r') as f:
        data = f.readlines()
    
        for line in data:
            odom = line.split()
            odom = int(odom[0])
            NodeIDList.append(odom)
    return NodeIDList

def ReadOrder(input_file_path):
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Order = []
    for chunk in reader:
        Order.append(chunk)
    Order = pd.concat(Order)
    Order = Order.drop(columns = ['End_time', 'PointS_Longitude', 'PointS_Latitude', 'PointE_Longitude', 'PointE_Latitude'])
    Order["Start_time"] = Order["Start_time"].apply(timestamp_datetime)
    Order = Order.sort_values(by = "Start_time")
    Order["ID"] = range(0,Order.shape[0])
    Order = Order.values
    return Order

def ReadResetOrder(input_file_path):
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Order = []
    for chunk in reader:
        Order.append(chunk)
    Order = pd.concat(Order)
    Order = Order.values
    return Order

def ReadDriver(input_file_path="./data/Drivers0601.csv"):
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Driver = []
    for chunk in reader:
        Driver.append(chunk)
    Driver = pd.concat(Driver)
    Driver = Driver.drop(columns=['Start_time'])
    Driver = Driver.values
    return Driver

def ReadAllFiles(OrderFileDate="0601"):
    NodePath = os.path.join(os.getcwd(),"data","Node.csv")
    NodeIDListPath = os.path.join(os.getcwd(),"data","NodeIDList.txt")
    OrdersPath = os.path.join(os.getcwd(),"data","order_2016" + OrderFileDate + ".csv")
    VehiclesPath = os.path.join(os.getcwd(),"data","Drivers0601.csv")
    MapPath = os.path.join(os.getcwd(),"data","AccurateMap.csv")

    Node = ReadNode(NodePath)
    NodeIDList = ReadNodeIDList(NodeIDListPath)
    Orders = ReadOrder(OrdersPath)
    Vehicles = ReadDriver(VehiclesPath)
    Map = ReadCostMap(MapPath)
    return Node,NodeIDList,Orders,Vehicles,Map

def ReadOrdersVehiclesFiles(OrderFileDate="0601"):
    OrdersPath = os.path.join(os.getcwd(),"data","order_2016" + OrderFileDate + ".csv")
    VehiclesPath = os.path.join(os.getcwd(),"data","Drivers0601.csv")
    Orders = ReadOrder(OrdersPath)
    Vehicles = ReadDriver(VehiclesPath)
    return Orders,Vehicles

def ReadLocalRegionBoundOrdersVehiclesFiles(SaveLocalRegionBoundOrdersPath):
    OrdersPath = os.path.join(os.getcwd(),"data","order_2016" + OrderFileDate + ".csv")
    Orders = ReadOrder(OrdersPath)
    return Orders

