
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

        self.InputSet = []
        self.GroundTruthSet = []

        self.Model = self.BuildModel(input_dim = len(train_X[0]), output_dim = len(train_Y[0]))


    def BuildModel(self):
        # 全连接神经网络
        Model = Sequential()
        #如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (samples, rows, cols, channels)
        Model.add(Conv2D(input_shape=(13,16,9),
        #Model.add(Conv2D(input_shape=(32,32,3),
                         filters=16,
                         kernel_size=5,
                         activation='relu',
                         padding='same'))

        #Model.add(MaxPooling2D(pool_size=(2, 2)))

        Model.add(Conv2D(filters=32,
                         kernel_size=3,
                         activation='relu',
                         padding='same'))


        Model.add(Conv2D(filters=1,
                         kernel_size=1,
                         activation='relu',
                         padding='same'))

        #Model.add(MaxPooling2D(pool_size=(2, 2)))

        #Model.add(Flatten())

        #Model.add(Dense(600, input_dim=input_dim, activation='relu'))
        #Model.add(Dense(600, activation='relu'))
        # Dropout层用于防止过拟合
        #Model.add(Dense(400, activation='relu'))
        #Model.add(Dense(300, activation='relu'))
        #Model.add(Dense(200, activation='relu'))
        #self.Model.add(Dropout(0.2))
        # 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
        #Model.add(Dense(208, activation='linear'))
        #Model.add(Dense(1, activation='linear'))
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
        early_stopping = EarlyStopping(monitor='val_loss', patience=5000, verbose=2)
        # 训练
        
        self.train_X = np.array(self.train_X)
        self.train_Y = np.array(self.train_Y)
        self.test_X = np.array(self.test_X)
        self.test_Y = np.array(self.test_Y)
        
        
        history = self.Model.fit(self.train_X, self.train_Y, epochs=20000,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2, shuffle=True, callbacks=[early_stopping])
        '''

        history = self.Model.fit(self.train_X, self.train_Y, epochs=300,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2, shuffle=True)
        '''
        self.Save("./model/PredictionModel.h5")
        # loss曲线
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()



class DeamdPredictionSimulationINFOCOM(Simulation):

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

        
    def CreateOrderDataSet(self):
        self.RealExpTime = self.Orders[0].ReleasTime
        EndTime = self.Orders[-1].ReleasTime
        self.NowOrder = self.Orders[0]
        self.step = 0

        StepGroundTruth = np.zeros(self.ClustersNumber)
        while self.RealExpTime <= EndTime:

            StepGroundTruth.fill(0)
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

        for i in range(len(self.GroundTruthSet)):
            if i == 0:
                continue

            self.InputSet.append(self.GroundTruthSet[i])

        del self.GroundTruthSet[-1]

        print(len(self.GroundTruthSet),len(self.GroundTruthSet[0]))
        print(len(self.InputSet),len(self.InputSet[0]))

        from sklearn.model_selection import train_test_split
        train_X, test_X, train_Y, test_Y = train_test_split(self.InputSet, self.GroundTruthSet, test_size=0.20)

        '''
        train_X = [x for y in train_X for x in y]
        train_Y = [x for y in train_Y for x in y]
        test_X = [x for y in test_X for x in y]
        test_Y = [x for y in test_Y for x in y]
        '''

        print(len(train_X),len(test_X))
        print(len(train_Y),len(test_Y))

        TestModel = PredictionModelINFOCOM(train_X=train_X, test_X=test_X, train_Y=train_Y, test_Y=test_Y)
        TestModel.Training()

        return




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

    EXP = DeamdPredictionSimulationINFOCOM(
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


