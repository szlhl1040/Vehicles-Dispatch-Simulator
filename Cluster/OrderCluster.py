import os
import math
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class OrderCluster(object):
        
    def __init__(self,n_clusters=None,NodeID=None,NodeLocation=None,NodeIDList=None,Orders=None,FocusOnLocalRegion=False,LocalRegionBound=None):
        self.n_clusters = n_clusters
        self.NodeID = NodeID
        self.NodeLocation = NodeLocation,
        self.NodeIDList = NodeIDList
        self.Orders = Orders
        self.labels_ = None

    def IsOrderInLimitRegion(self,OrderPickupNodeLocation):
        if OrderPickupNodeLocation[0] < self.LocalRegionBound[0] or OrderPickupNodeLocation[0] > self.LocalRegionBound[1]:
            return False
        elif OrderPickupNodeLocation[1] < self.LocalRegionBound[2] or OrderPickupNodeLocation[1] > self.LocalRegionBound[3]:
            return False
        else:
            return True

    def standardization(self,data):
        mu = np.mean(data)
        sigma = np.std(data)
        return (data - mu) / sigma

    def normalization(self,data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range


    def fit(self):
        LegalOrders = []

        for i in range(len(self.Orders)):
            if self.Orders[i][2] in self.NodeID and self.Orders[i][3] in self.NodeID:
                LegalOrders.append(self.Orders[i].tolist())

        NodeOrdersNum = np.zeros(len(self.NodeID))

        for i in range(len(LegalOrders)):
            NodeOrdersNum[self.NodeID.index(LegalOrders[i][2])] += 1

        self.NodeLocation = self.NodeLocation[0].tolist()
        for i in range(len(self.NodeLocation)):
            self.NodeLocation[i].append(int(NodeOrdersNum[i]))

        #print(self.NodeLocation)

        self.NodeLocation = self.normalization(self.NodeLocation[:])

        print(self.NodeLocation)

        estimator = KMeans(n_clusters=self.n_clusters)  # 构造聚类器
        estimator.fit(self.NodeLocation)  # 聚类

        self.labels_ = estimator.labels_

        return

