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

from matplotlib import pyplot
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam

from simulator.CNNDeamdPrediction import CNNPredictionModel

class DeamdPredictionSimulation(Simulation):

    def DefiningLocalVariables(self,epochs,PredictionModel_learning_rate,ReadOrdersNumber):
        self.epochs = epochs
        self.PredictionModel_learning_rate = PredictionModel_learning_rate
        self.ReadOrdersNumber = ReadOrdersNumber


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


        SinDayOfWeek = np.zeros(self.ClustersNumber)
        CosDayOfWeek = np.zeros(self.ClustersNumber)
        SinHourOfDay = np.zeros(self.ClustersNumber)
        CosHourOfDay = np.zeros(self.ClustersNumber)
        SinMinuteOfHour = np.zeros(self.ClustersNumber)
        CosMinuteOfHour = np.zeros(self.ClustersNumber)
        Weather = np.zeros(self.ClustersNumber)

        while self.RealExpTime <= EndTime:

            TimeWeatherState = self.GetTimeWeatherState(self.NowOrder)

            #TimeWeatherState[1] = day of week   
            SinDayOfWeek.fill(np.sin(TimeWeatherState[1]))
            CosDayOfWeek.fill(np.cos(TimeWeatherState[1]))
            #TimeWeatherState[2] = hour of day
            SinHourOfDay.fill(np.sin(TimeWeatherState[2]))
            CosHourOfDay.fill(np.cos(TimeWeatherState[2]))
            #TimeWeatherState[2] = minute of hour
            SinMinuteOfHour.fill(np.sin(TimeWeatherState[3]))
            CosMinuteOfHour.fill(np.cos(TimeWeatherState[3]))
            #Weather
            WeatherType = TimeWeatherState[4]/4.0
            #Weather.fill(TimeWeatherState[4])
            Weather.fill(WeatherType)

            StepPredictionRawData = np.zeros(self.ClustersNumber)

            #get one step's groundtruth
            #-------------------------------
            while self.NowOrder.ReleasTime < self.RealExpTime :
                if self.NowOrder.ID == self.Orders[-1].ID:
                    break

                NowCluster = self.NodeID2Cluseter[self.NowOrder.PickupPoint]
                StepPredictionRawData[NowCluster.ID] += 1

                self.NowOrder = self.Orders[self.NowOrder.ID+1]
            #-------------------------------

            self.PredictionRawData.append([StepPredictionRawData,
                                           SinDayOfWeek,
                                           CosDayOfWeek,
                                           SinHourOfDay,
                                           CosHourOfDay,
                                           SinMinuteOfHour,
                                           CosMinuteOfHour,
                                           Weather])

            self.step += 1
            self.RealExpTime += self.TimePeriods

        return


    def TransformFormat(self,RawData):
        Dimension = len(RawData)
        Res = []

        for i in range(len(RawData[0])):

            TempRes = np.zeros(Dimension)

            for j in range(Dimension):
                TempRes[j] = RawData[j][i]

            Res.append(TempRes)
        #for i in range(len(RawData[0])):
        Res = np.array(Res)

        Res = Res.reshape((self.NumGrideWidth,self.NumGrideHeight,Dimension))

        return Res


    def TrainingPretreatment(self):

        self.PredictionRawData = []

        for i in range(1,self.ReadOrdersNumber):
            if i < 10:
                OrderStr = "110" + str(i)
            else:
                OrderStr = "11" + str(i)

            self.Reset(OrderStr)
            self.CreateOrderDataSet()


        self.InputSet = []
        self.GroundTruthSet = []

        for i in range(len(self.PredictionRawData)):
            if i == 0:
                continue
            elif i == 1:
                #get Ground Truth
                self.GroundTruthSet.append(self.PredictionRawData[i][0])
            else:
                #get Ground Truth
                self.GroundTruthSet.append(self.PredictionRawData[i][0])

                self.InputSet.append(self.TransformFormat([self.PredictionRawData[i-2][0],
                                                           self.PredictionRawData[i-1][0],
                                                           self.PredictionRawData[i-1][1],
                                                           self.PredictionRawData[i-1][2],
                                                           self.PredictionRawData[i-1][3],
                                                           self.PredictionRawData[i-1][4],
                                                           self.PredictionRawData[i-1][5],
                                                           self.PredictionRawData[i-1][6],
                                                           self.PredictionRawData[i-1][7]]))

        del self.GroundTruthSet[-1]

        self.GroundTruthSet = np.array(self.GroundTruthSet)
        self.InputSet = np.array(self.InputSet)


        print("GroundTruthSet shape:",self.GroundTruthSet.shape)
        print("InputSet shape:",self.InputSet.shape)


        from sklearn.model_selection import train_test_split
        train_X, test_X, train_Y, test_Y = train_test_split(self.InputSet, self.GroundTruthSet, test_size=0.20)

        #train_X = [x for y in train_X for x in y]
        #train_Y = [x for y in train_Y for x in y]
        #test_X = [x for y in test_X for x in y]
        #test_Y = [x for y in test_Y for x in y]

        TestModel = CNNPredictionModel(PredictionModel_learning_rate=self.PredictionModel_learning_rate,
                                       NumGrideWidth=train_X[0].shape[0],
                                       NumGrideHeight=train_X[0].shape[1],
                                       NumGrideDimension=train_X[0].shape[2],
                                       OutputDimension=train_Y[0].shape[0],
                                       SideLengthMeter=self.SideLengthMeter,
                                       LocalRegionBound=self.LocalRegionBound,
                                       train_X=train_X,test_X=test_X,
                                       train_Y=train_Y,test_Y=test_Y)

        #TestModel.ReadData(train_X=train_X,test_X=test_X,train_Y=train_Y,test_Y=test_Y)

        TestModel.Training(self.epochs)

        '''
        print("Example for predict:")
        test_X = np.array(test_X[20:40])
        test_Y = np.array(test_Y[20:40])
        print(test_X.shape)
        print(test_Y.shape)
        print(test_X[0].shape)
        print(test_Y[0].shape)
        res = TestModel.Model.predict(test_X)
        for i in range(20):
            print('predict:',res[i].tolist())
            print('groundthruth',test_Y[i].tolist())
            print('-------------------')
        '''

        return


if __name__ == "__main__":

    explog = Logger()

    p2pConnectedThreshold = 0.8
    #ClusterMode = "KmeansClustering"
    #ClusterMode = "SpectralClustering"
    ClusterMode = "Grid"
    DeamdPredictionMode = "CNN"
    RebalanceMode = "Simulation"

    VehiclesNumber = 8000

    #500m
    SideLengthMeter = 800
    #1500m
    VehiclesServiceMeter = 2000

    LocalRegionBound = (104.035,104.105,30.625,30.695)

    #if FocusOnLocalRegion == False:
    #    LocalRegionBound = (104.007, 104.13, 30.6119, 30.7092)

    #FocusOnLocalRegion = True
    #FocusOnLocalRegion = True

    ReadOrdersNumber = 30
    PredictionModel_learning_rate = 0.00005
    epochs = 80000
    #train_X = None


    '''
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print(X_train.shape)
    print(X_test.shape)

    print(X_train[0])
    print(X_test[0])


    TestModel = PredictionModel(train_X=X_train, test_X=X_test, train_Y=y_train, test_Y=y_test)
    TestModel.Training()

    exit()
    '''


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

    EXP = DeamdPredictionSimulation(
                                    explog = explog,
                                    p2pConnectedThreshold = p2pConnectedThreshold,
                                    ClusterMode = ClusterMode,
                                    DeamdPredictionMode = DeamdPredictionMode,
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

    EXP.DefiningLocalVariables(epochs,PredictionModel_learning_rate,ReadOrdersNumber)
    EXP.CreateAllInstantiate()
    EXP.TrainingPretreatment()


