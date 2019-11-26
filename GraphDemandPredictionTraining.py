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

from simulator.utils import *
from simulator.simulator import Logger,Simulation
from simulator.GraphDemandPrediction import GraphConvolution,GraphPredictionModel


class DemandPredictionSimulation(Simulation):

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


    def CreateAdjacencyMatrix(self):
        self.AdjacencyMatrix = np.zeros((self.ClustersNumber,self.ClustersNumber))

        for i in self.Clusters:
            for j in i.Neighbor:
                self.AdjacencyMatrix[i.ID][j.ID] = 1
                self.AdjacencyMatrix[j.ID][i.ID] = 1

        from scipy import sparse
        self.AdjacencyMatrix = sparse.csr_matrix(self.AdjacencyMatrix)

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

                '''
                X = list(zip(self.PredictionRawData[i-2][0],
                             self.PredictionRawData[i-1][0],
                             self.PredictionRawData[i-1][1],
                             self.PredictionRawData[i-1][2],
                             self.PredictionRawData[i-1][3],
                             self.PredictionRawData[i-1][4],
                             self.PredictionRawData[i-1][5],
                             self.PredictionRawData[i-1][6],
                             self.PredictionRawData[i-1][7]))
                '''

                #Normalize X
                #X /= X.sum(1).reshape(-1, 1)

                #Graph = [X,A]
                self.InputSet.append([X,self.AdjacencyMatrix])

        del self.GroundTruthSet[-1]

        temp = self.GroundTruthSet[:]
        self.GroundTruthSet.clear()
        for i in temp:
            self.GroundTruthSet.append([])
            for j in i:
                self.GroundTruthSet[-1].append([j])
        del temp
        self.GroundTruthSet = np.array(self.GroundTruthSet)

        print("GroundTruthSet shape:",self.GroundTruthSet.shape)
        #print("InputSet shape:",self.InputSet.shape)

        return


    def CreateGraphNetwork(self):
        self.DemandPrediction = GraphPredictionModel(PredictionModel_learning_rate=self.PredictionModel_learning_rate,
                                                     ClustersNumber=self.ClustersNumber,
                                                     ClusterDimension=self.InputSet[0][0].shape[1],
                                                     OutputDimension=1,
                                                     SideLengthMeter=self.SideLengthMeter,
                                                     LocalRegionBound=self.LocalRegionBound)

        return


    def Training(self):
        '''
        print("input set ",type(self.InputSet))
        print("graph ",type(self.InputSet[0]))
        print("X ",type(self.InputSet[0][0]))
        print("A ",type(self.InputSet[0][1]))
        print("Ground Truth Set ",type(self.GroundTruthSet))
        print("Ground Truth",type(self.GroundTruthSet[0]))
        print()
        print("X shape ",self.InputSet[0][0].shape)
        print("A shape ",self.InputSet[0][1].shape)
        print("Ground Truth shape ",self.GroundTruthSet[0].shape)
        '''

        from sklearn.model_selection import train_test_split
        train_X, test_X, train_Y, test_Y = train_test_split(self.InputSet, self.GroundTruthSet, test_size=0.05)

        wait = 0
        preds = np.array([])
        best_val_loss = 999999
        PATIENCE = 250

        #for i in tqdm(range(self.epochs)):
        for i in range(self.epochs):

            print("step :",i)
            sumhistory = 0
            for j in range(len(train_X)):
                
                RandomTestNumber = random.randrange(len(test_X))
                history = self.DemandPrediction.Model.fit(train_X[j], train_Y[j], epochs=1,
                                                          validation_data=(test_X[RandomTestNumber], test_Y[RandomTestNumber]),
                                                          batch_size=self.ClustersNumber, verbose=0,shuffle=False)
                

                '''
                history = self.DemandPrediction.Model.fit(train_X[j], train_Y[j], epochs=1,
                                                          batch_size=self.ClustersNumber, verbose=0,shuffle=False)
                '''


                sumhistory += history.history['val_loss'][0]

            averagehistory = sumhistory/len(train_X)

            print("averagehistory val loss",averagehistory)

            self.Example(ExampleNumber=1)

            # Early stopping
            if averagehistory < best_val_loss:
                best_val_loss = averagehistory
                wait = 0
            else:
                if wait >= PATIENCE:
                    print('Epoch {}: early stopping'.format(i))
                    break
                wait += 1


        self.DemandPrediction.Save("./model/GCNDemandPredictionModel"+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".h5")
        
        return


    def Example(self,ExampleNumber):
        self.InputSet = self.InputSet[:ExampleNumber]
        self.GroundTruthSet = self.GroundTruthSet[:ExampleNumber]

        print("Example for predict:") 
        for i in range(len(self.InputSet)):

            res = self.DemandPrediction.Model.predict(self.InputSet[i], batch_size=self.ClustersNumber)
            #print('input:',self.InputSet[i])
            print('predict:',res.tolist())
            print('groundthruth:',self.GroundTruthSet[i].tolist())
            print('-------------------')

        return


    def Load(self):
        self.DemandPrediction.Load("./model/GCNDemandPredictionModel"+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".h5")

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

    ReadOrdersNumber = 5
    PredictionModel_learning_rate = 0.001
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
    #A
    EXP.CreateAdjacencyMatrix()
    EXP.CreateTrainingData()
    EXP.CreateGraphNetwork()
    EXP.Load()
    EXP.Training()
    #EXP.Example(ExampleNumber=10)
