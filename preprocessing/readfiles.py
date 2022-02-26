import os
import sys
from typing import Tuple, List
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime


def timestamp_datetime(value) -> datetime:
    d = datetime.fromtimestamp(value)
    t = dt.datetime(d.year,d.month,d.day,d.hour,d.minute,0)
    return t

def string_datetime(value):
    return dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

def string_pd_timestamp(value):
    d = string_datetime(value)
    t = pd.Timestamp(d.year, d.month, d.day, d.hour, d.minute)
    return t

def read_map(input_file_path: str) -> np.ndarray:
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    map_list = []
    for chunk in reader:
        map_list.append(chunk)
    map_df: pd.DataFrame = pd.concat(map_list)
    map_df = map_df.drop(["Unnamed: 0"], axis=1)
    map_values = map_df.values
    map_values = map_values.astype('int64')

    return map_values

def read_cost_map(input_file_path: str) -> pd.DataFrame:
    reader = pd.read_csv(input_file_path,header=None,chunksize = 1000)
    map_list = []
    for chunk in reader:
        map_list.append(chunk)
    map_df: pd.DataFrame = pd.concat(map_list)
    return map_df

def read_path(input_file_path) -> np.ndarray:
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    path_list = []
    for chunk in reader:
        path_list.append(chunk)
    path_df = pd.concat(path_list)
    path_df = path_df.drop(["Unnamed: 0"], axis=1)
    path_values = path_df.values
    path_values = path_values.astype('int64')
    return path_values

def read_node(input_file_path) -> pd.DataFrame:
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    node_list = []
    for chunk in reader:
        node_list.append(chunk)
    node_df = pd.concat(node_list)
    return node_df

def read_node_id_list(input_file_path) -> List[int]:
    node_id_list = []
    with open(input_file_path, 'r') as f:
        data = f.readlines()
    
        for line in data:
            odom = line.split()
            odom = int(odom[0])
            node_id_list.append(odom)
    return node_id_list

def read_order(input_file_path) -> pd.DataFrame:
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    order_list = []
    for chunk in reader:
        order_list.append(chunk)
    order_df: pd.DataFrame = pd.concat(order_list)
    order_df = order_df.drop(columns = ['End_time', 'PointS_Longitude', 'PointS_Latitude', 'PointE_Longitude', 'PointE_Latitude'])
    order_df["Start_time"] = order_df["Start_time"].apply(timestamp_datetime)
    order_df = order_df.sort_values(by = "Start_time")
    order_df["ID"] = range(0,order_df.shape[0])
    order_df = order_df[["ID", "Start_time", "NodeS", "NodeE"]]

    return order_df

def read_reset_order(input_file_path) -> np.ndarray:
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    order_list = []
    for chunk in reader:
        order_list.append(chunk)
    order_df = pd.concat(order_list)
    order_values = order_df.values
    return order_values

def read_driver(input_file_path="./data/Drivers0601.csv") -> np.ndarray:
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    driver_list = []
    for chunk in reader:
        driver_list.append(chunk)
    driver_df: pd.DataFrame = pd.concat(driver_list)
    if "Start_time" in driver_df.columns:
        driver_df = driver_df.drop(columns=['Start_time'])
    driver_values = driver_df.values
    return driver_values

def read_all_files(order_file_date: str = "0601") -> Tuple[pd.DataFrame, List[int], pd.DataFrame, np.ndarray, pd.DataFrame]:
    node_path = os.path.join(os.getcwd(),"data","Node.csv")
    node_id_list_path = os.path.join(os.getcwd(),"data","NodeIDList.txt")
    orders_path = os.path.join(os.getcwd(), "data", "Order", "modified", "train", "order_2016" + order_file_date + ".csv")
    vehicles_path = os.path.join(os.getcwd(),"data","Drivers0601.csv")
    map_path = os.path.join(os.getcwd(),"data","AccurateMap.csv")

    node_df = read_node(node_path)
    node_id_list = read_node_id_list(node_id_list_path)
    orders = read_order(orders_path)
    vehicles = read_driver(vehicles_path)
    map_df = read_cost_map(map_path)
    return node_df, node_id_list, orders, vehicles, map_df

def read_orders_vehicles_files(order_file_date: str = "0601"):
    orders_path = os.path.join(os.getcwd(),"data", "Order", "modified", "train", "order_2016" + order_file_date + ".csv")
    vehicles_path = os.path.join(os.getcwd(),"data","Drivers0601.csv")
    orders = read_order(orders_path)
    vehicles = read_driver(vehicles_path)
    return orders, vehicles

def read_local_region_bound_orders_vehicles_files(SaveLocalRegionBoundOrdersPath):
    OrdersPath = os.path.join(os.getcwd(),"data","order_2016" + order_file_date + ".csv")
    orders = read_order(OrdersPath)
    return orders

