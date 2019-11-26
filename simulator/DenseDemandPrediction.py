import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

class DensePredictionModel(object):
    #keras实现神经网络回归模型
    def __init__(self,PredictionModel_learning_rate,InputDimension,
                 OutputDimension,SideLengthMeter,LocalRegionBound):

        self.PredictionModel_learning_rate = PredictionModel_learning_rate
        self.InputDimension = InputDimension
        self.OutputDimension = OutputDimension
        self.SideLengthMeter = SideLengthMeter
        self.LocalRegionBound = LocalRegionBound

        self.InputSet = []
        self.GroundTruthSet = []

        self.Model = self.BuildModel()


    def BuildModel(self):
        Model = Sequential()
        Model.add(Dense(400, input_dim=self.InputDimension, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(800, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(800, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(800, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(800, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(400, activation='relu'))
        Model.add(Dropout(0.5))
        Model.add(Dense(self.OutputDimension, activation='linear'))
        Model.compile(loss='mse', optimizer=Adam(lr=self.PredictionModel_learning_rate))
        Model.summary()

        return Model

    def predict(self,X):
        return self.Model.predict(X)

    def Save(self, path):
        print("save CNN Demand Prediction model")
        self.Model.save_weights(path)

    def Load(self, path):
        print("load CNN Demand Prediction model")
        self.Model.load_weights(path)
