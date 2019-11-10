'''
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn import ensemble
from sklearn import ensemble
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
#from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt



    def PredictionMethod(var):

        return {
            # 方法选择
            # 1.决策树回归
            'model_decision_tree_regression' : tree.DecisionTreeRegressor(),
             
            # 2.线性回归
            'model_linear_regression' : LinearRegression(),
             
            # 3.SVM回归
            'model_svm' : svm.SVR(),
             
            # 4.kNN回归
            'model_k_neighbor' : neighbors.KNeighborsRegressor(),
             
            # 5.随机森林回归
            'model_random_forest_regressor' : ensemble.RandomForestRegressor(n_estimators=150),

            # 6.Adaboost回归
            'model_adaboost_regressor' : ensemble.AdaBoostRegressor(n_estimators=50),
             
            # 7.GBRT回归
            'model_gradient_boosting_regressor' : ensemble.GradientBoostingRegressor(n_estimators=100),
             
            # 8.Bagging回归
            'model_bagging_regressor' : ensemble.BaggingRegressor(),
             
            # 9.ExtraTree极端随机数回归
            'model_extra_tree_regressor' : ExtraTreeRegressor(),
        }.get(var,'error')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


class DeamdPrediction(object):
    """docstring for DeamdPrediction"""
    def __init__(self,Grids,NumGrideWidth=6,NumGrideHeight=5):
        self.Grids = Grids
        self.NumGrideWidth = NumGrideWidth
        self.NumGrideHeight = NumGrideHeight
        self.GridsHeatMap = self.CreateGridsHeatMap()
        self.DeamdPredictionGridsHeatMap = None

    def CreateGridsHeatMap(self,show=True):
        GridsHeatMap = np.zeros((self.NumGrideWidth,self.NumGrideHeight))
        GridsHeatMap = GridsHeatMap.tolist()
        #for i in reversed(range(self.NumGrideWidth)):
        for i in range(self.NumGrideWidth):
            for j in range(self.NumGrideHeight):
                GridsHeatMap[i][j] = len(self.Grids[i*self.NumGrideHeight + j].Orders)

        GridsHeatMap.reverse()
        GridsHeatMap = np.array(GridsHeatMap)

        print(GridsHeatMap)

        if show == True:
            seaborn.heatmap(GridsHeatMap, cmap='Reds')
            plt.show()

        return GridsHeatMap

    def PrintGridsHeatMap(self):
        return
'''


import os
import sys
import random
import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import numpy as np
import logging
import errno
import datetime as dt

from datetime import datetime,timedelta
from collections import deque
from objects.objects import Cluster,Order,Vehicle,Agent,Grid

from simulator.simulator import Logger,Simulation
from Cluster.TransportationCluster import TransportationCluster
from Cluster.OrderCluster import OrderCluster
from Cluster.SpectralCluster import SpectralCluster

from config.setting import *
from preprocessing.readfiles import *

from tqdm import tqdm
from matplotlib.pyplot import plot,savefig
from sklearn.cluster import KMeans






from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam



class PredictionModelINFOCOM(object):
    #keras实现神经网络回归模型
    def __init__(self,train_X,test_X,train_Y,test_Y):
        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_Y
        self.test_Y = test_Y

        self.Model = self.BuildModel()


    def BuildModel(self):
        # 全连接神经网络
        Model = Sequential()
        # input_shape=(self.X.shape[1],),
        Model.add(Dense(600, input_dim=len(self.train_X[0]), activation='relu'))
        # Dropout层用于防止过拟合
        Model.add(Dense(400, activation='relu'))
        Model.add(Dense(200, activation='relu'))
        Model.add(Dense(200, activation='relu'))
        Model.add(Dense(100, activation='relu'))
        #self.Model.add(Dropout(0.2))
        # 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
        Model.add(Dense(1))
        # 使用高效的 ADAM 优化算法以及优化的最小均方误差损失函数
        Model.compile(loss='mean_squared_error', optimizer=Adam())
        Model.summary()

        return Model


    def Save(self, path):
        print("save model")
        self.Model.save_weights(path)


    def Training(self):

        # early stoppping
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        # 训练
        
        self.train_X = np.array(self.train_X)
        self.train_Y = np.array(self.train_Y)
        self.test_X = np.array(self.test_X)
        self.test_Y = np.array(self.test_Y)
        '''
        history = self.Model.fit(self.train_X, self.train_Y, epochs=300,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2, shuffle=True, callbacks=[early_stopping])
        '''
        history = self.Model.fit(self.train_X, self.train_Y, epochs=300,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2)
        self.Save("./model/PredictionModel.h5")
        # loss曲线
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()



class DeamdPredictionINFOCOM(Simulation):

    def CreateGridsHeatMap(self,show=False):
        GridsHeatMap = np.zeros((self.NumGrideWidth,self.NumGrideHeight))
        GridsHeatMap = GridsHeatMap.tolist()
        for i in range(self.NumGrideWidth):
            for j in range(self.NumGrideHeight):
                GridsHeatMap[i][j] = len(self.Grids[i*self.NumGrideHeight + j].Orders)

        GridsHeatMap.reverse()
        GridsHeatMap = np.array(GridsHeatMap)

        if show:
            seaborn.heatmap(GridsHeatMap, cmap='Reds')
            plt.show()

        return GridsHeatMap


    def Reset(self,OrderFileDate="1101"):
        self.explog.LogInfo("Reset environment" + OrderFileDate)

        #过程变量
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None

        for i in self.Clusters:
            i.Reset()

        Orders,__ = ReadOrdersVehiclesFiles(OrderFileDate)

        self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]

        #限定订单发生区域
        #-------------------------------
        if self.FocusOnLocalRegion == True:
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        #-------------------------------

        return



    def Int2Bin(self, Int):
        BinStr = bin(Int).replace('0b','')

        BinList = []
        for i in BinStr:
            BinList.append(int(i))
        for i in range(64):
            if len(BinList) == ClusterIDBinarySize :
                break
            elif len(BinList) > ClusterIDBinarySize:
                print("Cluster ID Binary Size Error!")
                exit()
            else :
                BinList.insert(0,0)

        BinList = map(float,BinList)

        return BinList
        

    def CreateOrderDataSet(self):
        self.RealExpTime = self.Orders[0].ReleasTime
        EndTime = self.Orders[-1].ReleasTime
        self.NowOrder = self.Orders[0]
        self.step = 0

        StepGroundTruth = np.zeros(self.ClustersNumber)
        StepTrain = []

        while self.RealExpTime <= EndTime:

            StepGroundTruth.fill(0)

            #get Input
            #-------------------------------
            #self.TimeAndWeatherOneHotSignal = False
            if self.TimeAndWeatherOneHotSignal:
                WeatherTime = self.GetTimeWeatherOneHotNormalizationState(self.NowOrder)
            else:
                WeatherTime = self.GetTimeWeatherState(self.NowOrder)

            StepTrain.clear()
            for i in self.Clusters:
                Input = WeatherTime[:]
                #add Cluster ID with one-hot
                #Input += self.GetOneHot(N=i.ID,Len=self.ClustersNumber)
                #add Cluster ID with Int2Bin
                Input += self.Int2Bin(Int=i.ID)

                StepTrain.append(Input)

            self.InputSet.append(StepTrain)
                #print(Input)
            #-------------------------------

            #get one step's groundtruth
            #-------------------------------
            while self.NowOrder.ReleasTime < self.RealExpTime :
                if self.NowOrder.ID == self.Orders[-1].ID:
                    break

                NowCluster = self.NodeID2Cluseter[self.NowOrder.PickupPoint]
                StepGroundTruth[NowCluster.ID] += 1

                self.NowOrder = self.Orders[self.NowOrder.ID+1]
            #-------------------------------
            
            self.GroundTruthSet.append(StepGroundTruth.tolist())

            self.step += 1
            self.RealExpTime += self.TimePeriods

        return


    def Main(self):

        self.InputSet = []
        #self.GroundTruthSet = np.array()
        self.GroundTruthSet = []
        
        for i in range(31):
            if i == 0:
                continue

            if i < 10:
                OrderStr = "110" + str(i)
            else:
                OrderStr = "11" + str(i)

            self.Reset(OrderStr)
            self.CreateOrderDataSet()

        #print(len(self.InputSet))
        #print(len(self.GroundTruthSet))

        from sklearn.model_selection import train_test_split
        train_X, test_X, train_Y, test_Y = train_test_split(self.InputSet, self.GroundTruthSet, test_size=0.00)

        train_X = [x for y in train_X for x in y]
        train_Y = [x for y in train_Y for x in y]
        test_X = [x for y in test_X for x in y]
        test_Y = [x for y in test_Y for x in y]

        TestModel = PredictionModel(train_X=train_X, test_X=test_X, train_Y=train_Y, test_Y=test_Y)
        TestModel.Training()

        return


class DeamdPrediction(Simulation):
    """docstring for DeamdPrediction"""

    '''
    def __init__(self,explog,ClusterMode,ClustersNumber,
                NumGrideWidth,NumGrideHeight,TimePeriods,
                LocalRegionBound,TimeAndWeatherOneHotSignal,
                FocusOnLocalRegion):

        #统计变量
        self.RejectNum = 0

        #数据变量
        self.Clusters = None
        self.Orders = None
        self.Map = None
        self.Node = None
        self.NodeIDList = None
        self.NodeID2Cluseter = {}
        self.NodeID2NodesLocation = {}
        #Weather Table
        #------------------------------------------
        #晴=0 多云=1 阴=2 小雨=3
        #一天分上下两个天气状态
        self.WeatherTable11 = [2,1,1,1,1,0,1,2,1,1,3,3,3,3,3,
                               3,3,0,0,0,2,1,1,1,1,0,1,0,1,1,
                               1,3,1,1,0,2,2,1,0,0,2,3,2,2,2,
                               1,2,2,2,1,0,0,2,2,2,1,2,1,1,1]

        self.WeatherTable12 = [2,3,2,2,1,0,0,1,0,0,0,0,0,0,0,
                               1,2,2,1,1,1,2,1,0,1,3,2,3,3,1,
                               2,2,1,0,1,1,2,2,2,3,0,0,0,1,2,
                               2,2,3,3,2,1,2,0,1,0,1,1,1,1,3,
                               2,2] 
        #------------------------------------------

        #参数变量
        self.explog = explog
        self.ClusterMode = ClusterMode
        self.ClustersNumber = ClustersNumber
        self.NumGrideWidth = NumGrideWidth
        self.NumGrideHeight = NumGrideHeight
        self.TimePeriods = TimePeriods 
        self.LocalRegionBound = LocalRegionBound    #(1,2,3,4) = (左,右,下,上)

        #控制变量
        self.TimeAndWeatherOneHotSignal = TimeAndWeatherOneHotSignal
        self.FocusOnLocalRegion = FocusOnLocalRegion

        #过程变量
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None
        self.Episode = 0


    def Reset(self,OrderFileDate="1101"):
        self.explog.LogInfo("Reset environment")

        self.Orders = None

        self.RealExpTime = None
        self.NowOrder = None
        self.step = None

        for i in self.Clusters:
            i.Reset()

        Orders,__ = ReadOrdersVehiclesFiles(OrderFileDate)

        self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]

        #限定订单发生区域
        #-------------------------------
        if self.FocusOnLocalRegion == True:
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        #-------------------------------

        return

    
    def CreateGrid(self):

        NumGrideHeight = self.NumGrideHeight
        NumGride = self.NumGrideWidth * self.NumGrideHeight

        NodeLocation = self.Node[['Longitude','Latitude']].values.round(7)
        NodeID = self.Node['NodeID'].values.astype('int')

        if self.FocusOnLocalRegion == True:
            NodeLocation = NodeLocation.tolist()
            NodeID = NodeID.tolist()

            TempNodeList = []
            for i in range(len(NodeLocation)):
                TempNodeList.append((NodeLocation[i],NodeID[i]))

            for i in TempNodeList[:]:
                if self.IsNodeInLimitRegion(i) == False:
                    TempNodeList.remove(i)

            NodeLocation.clear()
            NodeID.clear()

            for i in TempNodeList:
                #NodeLocation.append(i[0])
                NodeLocation.append(i[0])
                NodeID.append(i[1])

            NodeLocation = np.array(NodeLocation)

        NodeSet = {}
        for i in range(len(NodeID)):
            NodeSet[(NodeLocation[i][0],NodeLocation[i][1])] = self.NodeIDList.index(NodeID[i])

        #装载Node进每一个Grid
        #------------------------------------------------------
        if self.FocusOnLocalRegion == True:
            TotalWidth = self.LocalRegionBound[1] - self.LocalRegionBound[0]
            TotalHeight = self.LocalRegionBound[3] - self.LocalRegionBound[2]
        else:
            TotalWidth = 104.13 - 104.00767
            TotalHeight = 30.7092 - 30.6119

        IntervalWidth = TotalWidth / self.NumGrideWidth
        IntervalHeight = TotalHeight / self.NumGrideHeight

        AllGrid = [Grid(i,[],[],{},[],{},[]) for i in range(NumGride)]

        for key,value in NodeSet.items():
            #print(key[0],key[1],value)

            NowGridWidthNum = None
            NowGridHeightNum = None

            # 问题在i
            for i in range(self.NumGrideWidth):
                if self.FocusOnLocalRegion == True:
                    LeftBound = (self.LocalRegionBound[0] + i * IntervalWidth)
                    RightBound = (self.LocalRegionBound[0] + (i+1) * IntervalWidth)
                else:
                    LeftBound = (104.007 + i * IntervalWidth)
                    RightBound = (104.007 + (i+1) * IntervalWidth)

                if key[0] > LeftBound and key[0] < RightBound:
                    NowGridWidthNum = i
                    break

            for i in range(self.NumGrideHeight):
                if self.FocusOnLocalRegion == True:
                    DownBound = (self.LocalRegionBound[2] + i * IntervalHeight)
                    UpBound = (self.LocalRegionBound[2] + (i+1) * IntervalHeight)
                else:
                    DownBound = (30.6119 + i * IntervalHeight)
                    UpBound = (30.6119 + (i+1) * IntervalHeight)

                if key[1] > DownBound and key[1] < UpBound:
                    NowGridHeightNum = i
                    break

            if NowGridWidthNum == None or NowGridHeightNum == None :
                print("error")
                print(key[0],key[1])
                exit()
            else:
                AllGrid[self.NumGrideWidth * NowGridHeightNum + NowGridWidthNum].Nodes.append((value,(key[0],key[1])))
        #------------------------------------------------------

        return AllGrid


    def CreateCluster(self):

        NodeLocation = self.Node[['Longitude','Latitude']].values.round(7)
        NodeID = self.Node['NodeID'].values.astype('int64')

        N = {}
        for i in range(len(NodeID)):
            N[(NodeLocation[i][0],NodeLocation[i][1])] = NodeID[i]

        Clusters=[Cluster(i,[],[],{},[],{},[]) for i in range(self.ClustersNumber)]

        #进行Kmeans聚类，如果有进行过，则不再重新进行
        #----------------------------------------------
        if self.FocusOnLocalRegion == True:
            SaveClusterPath = './data/'+str(self.LocalRegionBound)+str(self.ClustersNumber)+str(self.ClusterMode)+'Label.csv'
        else:
            SaveClusterPath = './data/'+str(self.ClustersNumber)+str(self.ClusterMode)+'Label.csv'
        if os.path.exists(SaveClusterPath):
            ClusterSignal = False
        else :
            ClusterSignal = True

        if ClusterSignal :
            print("No Cluster Data")
            print("Please run Rebalance Agent to create cluster or grid data")
            exit()
        else:
            self.explog.LogInfo("Loading Cluster results")
            reader = pd.read_csv(SaveClusterPath,chunksize = 1000)
            label_pred = []
            for chunk in reader:
                label_pred.append(chunk)
            label_pred = pd.concat(label_pred)
            label_pred = label_pred.values
            label_pred = label_pred.flatten()
            label_pred = label_pred.astype('int')
        #----------------------------------------------
        #将聚类数据载入到实例化的类里
        for i in range(self.ClustersNumber):
            temp = NodeLocation[label_pred == i]
            for j in range(len(temp)):
                Clusters[i].Nodes.append((self.NodeIDList.index(N[(temp[j,0],temp[j,1])]),(temp[j,0],temp[j,1])))

        return Clusters

    def CreateAllInstantiate(self,OrderFileDate="1101"):
        self.explog.LogInfo("Read all files")
        self.Node,__,self.NodeIDList,Orders,__,self.Map = ReadAllFiles(OrderFileDate)

        #if self.ClusterMode == "KmeansCluster" or self.ClusterMode == "TransportationCluster":
        if self.ClusterMode != "Grid":
            self.explog.LogInfo("Create Clusters")
            self.Clusters = self.CreateCluster()
        elif self.ClusterMode == "Grid":
            self.explog.LogInfo("Create Grids")
            self.Clusters = self.CreateGrid()

        self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]

        #限定订单发生区域
        #-------------------------------
        if self.FocusOnLocalRegion == True:
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        #-------------------------------

        NodeID = self.Node['NodeID'].values

        for i in range(len(NodeID)):
            NodeID[i] = self.NodeIDList.index(NodeID[i])

        #self.NodeID2Cluseter = {}
        for i in NodeID:
            for j in self.Clusters:
                for k in j.Nodes:
                    #print(i,type(i),self.NodeIDList.index(i),type(self.NodeIDList.index(i)),k,type(k),k[0],type(k[0]),j,type(j))
                    if i == k[0]:
                        self.NodeID2Cluseter[i] = j
        return
    '''
    def Reset(self,OrderFileDate="1101"):
        self.explog.LogInfo("Reset environment" + OrderFileDate)

        #过程变量
        self.RealExpTime = None
        self.NowOrder = None
        self.step = None

        for i in self.Clusters:
            i.Reset()

        Orders,__ = ReadOrdersVehiclesFiles(OrderFileDate)

        self.Orders = [Order(i[0],i[1],self.NodeIDList.index(i[2]),self.NodeIDList.index(i[3]),i[1]+PICKUPTIMEWINDOW,None,None,None) for i in Orders]

        #限定订单发生区域
        #-------------------------------
        if self.FocusOnLocalRegion == True:
            for i in self.Orders[:]:
                if self.IsOrderInLimitRegion(i) == False:
                    self.Orders.remove(i)
            for i in range(len(self.Orders)):
                self.Orders[i].ID = i
        #-------------------------------

        return



    def Int2Bin(self, Int):
        BinStr = bin(Int).replace('0b','')

        BinList = []
        for i in BinStr:
            BinList.append(int(i))
        for i in range(64):
            if len(BinList) == ClusterIDBinarySize :
                break
            elif len(BinList) > ClusterIDBinarySize:
                print("Cluster ID Binary Size Error!")
                exit()
            else :
                BinList.insert(0,0)

        BinList = map(float,BinList)

        return BinList
        

    def CreateOrderDataSet(self):
        self.RealExpTime = self.Orders[0].ReleasTime
        EndTime = self.Orders[-1].ReleasTime
        self.NowOrder = self.Orders[0]
        self.step = 0

        StepGroundTruth = np.zeros(self.ClustersNumber)
        StepTrain = []

        while self.RealExpTime <= EndTime:

            StepGroundTruth.fill(0)

            #get Input
            #-------------------------------
            self.TimeAndWeatherOneHotSignal = False
            if self.TimeAndWeatherOneHotSignal:
                WeatherTime = self.GetTimeWeatherOneHotNormalizationState(self.NowOrder)
            else:
                WeatherTime = self.GetTimeWeatherState(self.NowOrder)

            StepTrain.clear()
            for i in self.Clusters:
                Input = WeatherTime[:]
                #add Cluster ID with one-hot
                #Input += self.GetOneHot(N=i.ID,Len=self.ClustersNumber)
                #add Cluster ID with Int2Bin
                Input += self.Int2Bin(Int=i.ID)

                StepTrain.append(Input)

            self.InputSet.append(StepTrain)
                #print(Input)
            #-------------------------------

            #get one step's groundtruth
            #-------------------------------
            while self.NowOrder.ReleasTime < self.RealExpTime :
                if self.NowOrder.ID == self.Orders[-1].ID:
                    break

                NowCluster = self.NodeID2Cluseter[self.NowOrder.PickupPoint]
                StepGroundTruth[NowCluster.ID] += 1

                self.NowOrder = self.Orders[self.NowOrder.ID+1]
            #-------------------------------
            
            self.GroundTruthSet.append(StepGroundTruth.tolist())

            self.step += 1
            self.RealExpTime += self.TimePeriods

        return


    def Main(self):

        self.InputSet = []
        #self.GroundTruthSet = np.array()
        self.GroundTruthSet = []
        
        for i in range(2):
            if i == 0:
                continue

            if i < 10:
                OrderStr = "110" + str(i)
            else:
                OrderStr = "11" + str(i)

            self.Reset(OrderStr)
            self.CreateOrderDataSet()

        #print(len(self.InputSet))
        #print(len(self.GroundTruthSet))

        self.NEW = []

        #每个时间段
        for i in range(len(self.GroundTruthSet)):
            #每个grid内的order数量
            for j in range(len(self.GroundTruthSet[0])):
                if i == 0:
                    pass
                else:
                    self.NEW.append(self.InputSet[i-1][j])
                    self.NEW[-1].append(self.GroundTruthSet[i-1][j])
                    #self.InputSet[i-1][j].append(self.GroundTruthSet[i-1][j])

        for j in range(len(self.GroundTruthSet[-1])):
            self.NEW.append(self.InputSet[-1][j])
            self.NEW[-1].append(self.GroundTruthSet[-1][j])
            #self.InputSet[-1][j].append(self.GroundTruthSet[-1][j])

        print(len(self.NEW))

        print(self.NEW)


        self.InputSet = [x for y in self.InputSet for x in y]
        self.GroundTruthSet = [x for y in self.GroundTruthSet for x in y]


        print(len(self.InputSet))
        print(len(self.GroundTruthSet))

        print(self.InputSet)

        exit()


        from sklearn.model_selection import train_test_split
        train_X, test_X, train_Y, test_Y = train_test_split(self.InputSet, self.GroundTruthSet, test_size=0.20)

        TestModel = PredictionModel(train_X=train_X, test_X=test_X, train_Y=train_Y, test_Y=test_Y)
        TestModel.Training()

        return


class PredictionModel(object):
    #keras实现神经网络回归模型
    def __init__(self,train_X,test_X,train_Y,test_Y):
        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_Y
        self.test_Y = test_Y

        self.Model = self.BuildModel()


    def BuildModel(self):
        # 全连接神经网络
        Model = Sequential()
        # input_shape=(self.X.shape[1],),
        Model.add(Dense(600, input_dim=len(self.train_X[0]), activation='relu'))
        # Dropout层用于防止过拟合
        Model.add(Dense(400, activation='relu'))
        Model.add(Dense(200, activation='relu'))
        Model.add(Dense(200, activation='relu'))
        Model.add(Dense(100, activation='relu'))
        #self.Model.add(Dropout(0.2))
        # 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
        Model.add(Dense(1))
        # 使用高效的 ADAM 优化算法以及优化的最小均方误差损失函数
        Model.compile(loss='mean_squared_error', optimizer=Adam())
        Model.summary()

        return Model


    def Save(self, path):
        print("save model")
        self.Model.save_weights(path)


    def Training(self):

        # early stoppping
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
        # 训练
        
        self.train_X = np.array(self.train_X)
        self.train_Y = np.array(self.train_Y)
        self.test_X = np.array(self.test_X)
        self.test_Y = np.array(self.test_Y)

        #print(len(self.train_X),len())
        #print(len(self.train_Y))
        '''
        history = self.Model.fit(self.train_X, self.train_Y, epochs=300,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2, shuffle=True, callbacks=[early_stopping])
        
        '''
        history = self.Model.fit(self.train_X, self.train_Y, epochs=300,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2)
        

        self.Save("./model/PredictionModel.h5")
        # loss曲线
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

        '''
        # 预测
        yhat = self.Model.predict(test_X)
        # 预测y逆标准化
        inv_yhat0 = concatenate((test_X, yhat), axis=1)
        inv_yhat1 = scaler.inverse_transform(inv_yhat0)
        inv_yhat = inv_yhat1[:,-1]
        # 原始y逆标准化
        test_y = test_y.reshape((len(test_y), 1))
        inv_y0 = concatenate((test_X,test_y), axis=1)
        inv_y1 = scaler.inverse_transform(inv_y0)
        inv_y = inv_y1[:,-1]
        # 计算 RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        '''


        '''
        plt.plot(inv_y)
        plt.plot(inv_yhat)
        plt.show()
        '''


if __name__ == "__main__":


    explog = Logger()
    p2pConnectedThreshold = 0.8
    #ClusterMode = "KmeansClustering"
    #ClusterMode = "SpectralClustering"
    ClusterMode = "Grid"
    RebalanceMode = "Simulation"

    VehiclesNumber = 3000

    #500m
    SideLengthMeter = 500
    #1500m
    VehiclesServiceMeter =2000

    LocalRegionBound = (104.035,104.105,30.625,30.695)

    FocusOnLocalRegion = True
    '''
    EXP = DeamdPrediction(
                            explog = explog,
                            ClusterMode = ClusterMode,
                            ClustersNumber = 30,
                            NumGrideWidth = 6,
                            NumGrideHeight = 5,
                            TimePeriods = TIMESTEP,
                            LocalRegionBound = LocalRegionBound,
                            TimeAndWeatherOneHotSignal = TimeAndWeatherOneHotSignal,
                            FocusOnLocalRegion = FocusOnLocalRegion,
                            )
    '''

    EXP = Simulation(
                    explog = explog,
                    p2pConnectedThreshold = p2pConnectedThreshold,
                    ClusterMode = ClusterMode,
                    RebalanceMode = RebalanceMode,
                    VehiclesNumber = VehiclesNumber,
                    TimePeriods = TIMESTEP,
                    LocalRegionBound = LocalRegionBound,
                    SideLengthMeter = SideLengthMeter,
                    VehiclesServiceMeter = VehiclesServiceMeter,
                    TimeAndWeatherOneHotSignal = TimeAndWeatherOneHotSignal,
                    NeighborCanServer = NeighborCanServer,
                    FocusOnLocalRegion = FocusOnLocalRegion,
                    SaveMemorySignal = SaveMemorySignal
                    )

    EXP.CreateAllInstantiate()

    EXP.Main()


