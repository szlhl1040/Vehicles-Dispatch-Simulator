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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding, LSTM
from keras.optimizers import Adam

class CNNPredictionModel(object):
    #keras实现神经网络回归模型
    def __init__(self,PredictionModel_learning_rate,NumGrideWidth,
                 NumGrideHeight,NumGrideDimension,OutputDimension,
                 SideLengthMeter,LocalRegionBound,train_X,test_X,
                 train_Y,test_Y):

        self.PredictionModel_learning_rate = PredictionModel_learning_rate
        self.NumGrideWidth = NumGrideWidth
        self.NumGrideHeight = NumGrideHeight
        self.NumGrideDimension = NumGrideDimension
        self.OutputDimension = OutputDimension
        self.SideLengthMeter = SideLengthMeter
        self.LocalRegionBound = LocalRegionBound

        self.InputSet = []
        self.GroundTruthSet = []

        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_Y
        self.test_Y = test_Y

        self.Model = self.BuildModel()

    def ReadData(train_X,test_X,train_Y,test_Y):
        self.train_X = train_X
        self.test_X = test_X
        self.train_Y = train_Y
        self.test_Y = test_Y

    def BuildModel(self):
        Model = Sequential()
        #如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (samples, rows, cols, channels)
        #print(self.NumGrideWidth,self.NumGrideHeight,self.NumGrideDimension,self.OutputDimension)
        #print(type(self.NumGrideWidth),type(self.NumGrideHeight),type(self.NumGrideDimension),type(self.OutputDimension))
        Model.add(Conv2D(input_shape=(self.NumGrideWidth,self.NumGrideHeight,self.NumGrideDimension),
                         filters=16,
                         kernel_size=5,
                         activation='relu',
                         padding='same'))
        Model.add(MaxPooling2D(pool_size=(2, 2)))

        Model.add(Conv2D(filters=32,
                         kernel_size=3,
                         activation='relu',
                         padding='same'))
        Model.add(MaxPooling2D(pool_size=(2, 2))) 

        Model.add(Flatten())
        #Model.add(Dense(600, input_dim=input_dim, activation='relu'))
        Model.add(Dense(800, activation='relu'))
        # Dropout层用于防止过拟合
        #Model.add(Dropout(0.2))
        Model.add(Dense(600, activation='relu'))
        Model.add(Dense(400, activation='relu'))
        Model.add(Dense(400, activation='relu'))
        # 没有激活函数用于输出层，因为这是一个回归问题，我们希望直接预测数值，而不需要采用激活函数进行变换。
        #Model.add(Dense(len(self.train_Y[0]), activation='linear'))
        Model.add(Dense(self.OutputDimension, activation='relu'))
        # 使用高效的 ADAM 优化算法以及优化的最小均方误差损失函数
        Model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.PredictionModel_learning_rate))
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


    def Training(self,epochs=1000):
        # early stoppping
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=250, verbose=2)
        # 训练

        self.train_X = np.array(self.train_X)
        self.train_Y = np.array(self.train_Y)
        self.test_X = np.array(self.test_X)
        self.test_Y = np.array(self.test_Y)

        print(self.train_X.shape,self.train_Y.shape)
        print(self.train_X[0].shape,self.train_Y[0].shape)

        '''
        print(self.train_X.shape,self.train_Y.shape)
        print(self.train_X[0].shape,self.train_Y[0].shape)

        print(self.train_X[0],self.train_Y[0])
        print(self.train_X[1],self.train_Y[1])
        '''
        
        history = self.Model.fit(self.train_X, self.train_Y, epochs=epochs,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2, shuffle=True, callbacks=[early_stopping])

        '''
        history = self.Model.fit(self.train_X, self.train_Y, epochs=epochs,
                                batch_size=32, validation_data=(self.test_X, self.test_Y),
                                verbose=2)
        '''

        self.Save("./model/GridCNN"+str(self.SideLengthMeter)+str(self.LocalRegionBound)+".h5")
        # loss曲线
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
