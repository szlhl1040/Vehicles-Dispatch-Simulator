import os
import math
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
#import evaluate as eval

class SpectralCluster(object):
        
    def __init__(self,n_clusters=None,Knn_k = 5,NodeID=None,NodeIDList=None,Map=None,FocusOnLocalRegion=False,LocalRegionBound=None):
        self.n_clusters = n_clusters
        self.Knn_k = Knn_k
        self.NodeID = NodeID
        self.NodeIDList = NodeIDList
        self.Map = Map
        self.DataMatrix = np.zeros((len(self.NodeID), len(self.NodeID)))
        self.FocusOnLocalRegion = FocusOnLocalRegion
        self.LocalRegionBound = LocalRegionBound    #(1,2,3,4) = (左,右,下,上)
        self.labels_ = None


    def getW(self):
        """
        利用KNN获得相似矩阵
        :param k: KNN参数
        :return:
        """
        W = np.zeros((len(self.NodeID), len(self.NodeID)))
        for idx, each in enumerate(self.DataMatrix):
            index_array = np.argsort(each)
            #W[idx][index_array[1:self.Knn_k+1]] = 1
            for i in index_array[1:self.Knn_k+1]:
                W[idx][i] = self.DataMatrix[idx][i]  # 距离最短的是自己
        tmp_W = np.transpose(W)
        W = (tmp_W+W)/2
        return W


    def getD(self,W):
        """
        获得度矩阵
        :param W:  相似度矩阵
        :return:   度矩阵
        """
        D = np.diag(sum(W))
        return D


    def getL(self,D,W):
        """
        获得拉普拉斯举着
        :param W: 相似度矩阵
        :param D: 度矩阵
        :return: 拉普拉斯矩阵
        """
        return D - W


    def getEigen(self,L):
        """
        从拉普拉斯矩阵获得特征矩阵
        :param L: 拉普拉斯矩阵
        :return:
        """
        eigval, eigvec = np.linalg.eig(L)
        ix = np.argsort(eigval)[0:self.n_clusters]
        return eigvec[:, ix]


    def RoadCost(self,start,end):
        return self.Map[start][end]


    def fit(self):
        #转换成4000的ID
        for i in range(len(self.NodeID)):
            self.NodeID[i] = self.NodeIDList.index(self.NodeID[i])

        for i in range(len(self.NodeID)):
            for j in range(len(self.NodeID)):
                self.DataMatrix[i][j] = self.RoadCost(self.NodeID[i],self.NodeID[j])

        W = self.getW()
        D = self.getD(W)
        L = self.getL(D, W)

        eigvec = self.getEigen(L)

        estimator = KMeans(n_clusters=self.n_clusters)
        estimator.fit(eigvec)
        self.labels_ = estimator.labels_

        return
