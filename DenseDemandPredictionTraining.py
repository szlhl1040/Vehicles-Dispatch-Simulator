import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

from matplotlib.pyplot import plot,savefig

from matplotlib import pyplot
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K

import random
from collections import deque

from simulator.simulator import Logger,Simulation
from Cluster.TransportationCluster import TransportationCluster
from Cluster.OrderCluster import OrderCluster
from Cluster.SpectralCluster import SpectralCluster

from config.setting import *
from preprocessing.readfiles import *

from tqdm import tqdm
from matplotlib.pyplot import plot,savefig
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam

from simulator.simulator import Logger,Simulation
from simulator.DenseDemandPrediction import DensePredictionModel


class DemandPredictionSimulation(Simulation):

    def DefiningLocalVariables(self,epochs,PredictionModel_learning_rate,ReadOrdersNumber):
        self.epochs = epochs
        self.PredictionModel_learning_rate = PredictionModel_learning_rate
        self.ReadOrdersNumber = ReadOrdersNumber


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

        return Res


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


    def CreateTrainingData(self):    
        self.PredictionRawData = []

        for i in range(1,self.ReadOrdersNumber):
            if i < 10:
                OrderStr = "110" + str(i)
            else:
                OrderStr = "11" + str(i)

            self.Reset(OrderStr)
            self.CreateOrderDataSet()

        #print(len(self.PredictionRawData))

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

                X = self.TransformFormat([self.PredictionRawData[i-2][0],
                                          self.PredictionRawData[i-1][0],
                                          self.PredictionRawData[i-1][1],
                                          self.PredictionRawData[i-1][2],
                                          self.PredictionRawData[i-1][3],
                                          self.PredictionRawData[i-1][4],
                                          self.PredictionRawData[i-1][5],
                                          self.PredictionRawData[i-1][6],
                                          self.PredictionRawData[i-1][7]])

                #Normalize X
                #X /= X.sum(1).reshape(-1, 1)

                #Graph = [X,A]
                self.InputSet.append(X)

        del self.GroundTruthSet[-1]

        self.GroundTruthSet = np.array(self.GroundTruthSet)
        self.InputSet = np.array(self.InputSet)
        
        #flatten input
        temp = []
        for i in self.InputSet:
            temp.append(i.flatten())

        self.InputSet = np.array(temp)

        print("InputSet shape:",self.InputSet.shape)
        print("GroundTruthSet shape:",self.GroundTruthSet.shape)

        return


    def CreateDenseNetwork(self):
        self.DemandPrediction = DensePredictionModel(PredictionModel_learning_rate=self.PredictionModel_learning_rate,
                                                     InputDimension = self.InputSet[0].shape[0],
                                                     OutputDimension = self.GroundTruthSet[0].shape[0],
                                                     SideLengthMeter=self.SideLengthMeter,
                                                     LocalRegionBound=self.LocalRegionBound)
        return


    def Training(self):

        from sklearn.model_selection import train_test_split
        train_X, test_X, train_Y, test_Y = train_test_split(self.InputSet, self.GroundTruthSet, test_size=0.10)

        # early stoppping
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=250, verbose=2)


        history = self.DemandPrediction.Model.fit(train_X, train_Y, epochs=epochs,batch_size=32,
                                                  validation_data=(test_X, test_Y),verbose=2,
                                                  shuffle=False, callbacks=[early_stopping])

        # loss曲线
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()


        self.DemandPrediction.Save("./model/DenseDemandPredictionModel"+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".h5")
        
        return


    def Example(self,ExampleNumber):
        self.InputSet = self.InputSet[:ExampleNumber]
        self.GroundTruthSet = self.GroundTruthSet[:ExampleNumber]

        for i in range(len(self.InputSet)):
            temp = []
            for j in self.InputSet:
                temp.append([j])

            res = self.DemandPrediction.Model.predict(temp)
            #print('input:',self.InputSet[i])
            print('predict:',res.tolist())
            print('groundthruth:',self.GroundTruthSet[i].tolist())
            print('-------------------')

        return


    def Load(self):
        self.DemandPrediction.Load("./model/DenseDemandPredictionModel"+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".h5")

        return


if __name__ == "__main__":

    explog = Logger()

    p2pConnectedThreshold = 0.8
    ClusterMode = "KmeansClustering"
    DemandPredictionMode = "GCN"
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

    ReadOrdersNumber = 31
    PredictionModel_learning_rate = 0.00005
    epochs = 50000

    EXP = DemandPredictionSimulation(
                                    explog = explog,
                                    p2pConnectedThreshold = p2pConnectedThreshold,
                                    ClusterMode = ClusterMode,
                                    DemandPredictionMode = DemandPredictionMode,
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
    EXP.CreateTrainingData()
    EXP.CreateDenseNetwork()
    EXP.Training()
    #EXP.Load()
    EXP.Example(ExampleNumber=10)
