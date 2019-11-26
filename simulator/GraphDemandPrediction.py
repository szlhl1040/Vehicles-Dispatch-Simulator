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
from keras.optimizers import Adam
from keras.layers import Input, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from simulator.utils import *



class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphPredictionModel(object):
    #keras实现神经网络回归模型
    def __init__(self,PredictionModel_learning_rate,ClustersNumber,
    			 ClusterDimension,OutputDimension,
                 SideLengthMeter,LocalRegionBound):

        self.PredictionModel_learning_rate = PredictionModel_learning_rate
        self.ClustersNumber = ClustersNumber
        self.ClusterDimension = ClusterDimension
        self.OutputDimension = OutputDimension
        self.SideLengthMeter = SideLengthMeter
        self.LocalRegionBound = LocalRegionBound

        self.Model = self.BuildModel()


    def BuildModel(self):
        print('Using local pooling filters...')
        support = 1
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
        X_in = Input(shape=(self.ClusterDimension,))
        # Define model architecture
        # NOTE: We pass arguments for graph convolutional layers as a list of tensors.
        # This is somewhat hacky, more elegant options would require rewriting the Layer base class.
        H = Dropout(0.1)(X_in)
        H = GraphConvolution(256, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
        #H = Dropout(0.5)(H)
        H = GraphConvolution(256, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
        #H = Dropout(0.5)(H)
        #H = GraphConvolution(256, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
        #Y = GraphConvolution(y.shape[1], support, activation='softmax')([H]+G)
        #Y = GraphConvolution(self.OutputDimension, support, activation='softmax')([H]+G)
        Y = GraphConvolution(self.OutputDimension, support, activation='linear')([H]+G)

        # Compile model
        model = Model(inputs=[X_in]+G, outputs=Y)
        #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.PredictionModel_learning_rate))
        model.summary()

        return model

    def predict(self,X):
        return self.Model.predict(X)

    def Save(self, path):
        print("save GCN Demand Prediction model")
        self.Model.save_weights(path)

    def Load(self, path):
        print("load GCN Demand Prediction model")
        self.Model.load_weights(path)

