import os
import math
import random
import pandas as pd
import numpy as np


class TransportationCluster(object):
        
    def __init__(self,n_clusters=None,NodeID=None,NodeLocation=None,NodeIDList=None,Map=None,NumGrideWidth=6,NumGrideHeight=5,FocusOnLocalRegion=False,LocalRegionBound=None):
        self.n_clusters = n_clusters
        self.NodeID = NodeID
        self.NodeLocation = NodeLocation
        self.NodeIDList = NodeIDList
        self.Map = Map
        self.NumGrideWidth = NumGrideWidth
        self.NumGrideHeight = NumGrideHeight
        self.FocusOnLocalRegion = FocusOnLocalRegion
        self.LocalRegionBound = LocalRegionBound    #(1,2,3,4) = (左,右,下,上)

        self.labels_ = []

        self.EARTH_REDIUS = 6378.137
        self.PI = 3.141592653589793
        self.ShortestNodeList = []

    def rad(self,d):
        return d * self.PI / 180.0

    def GetDistance(self,longitude1,latitude1,longitude2,latitude2):
        radLat1 = self.rad(latitude1)
        radLat2 = self.rad(latitude2)
        a = radLat1 - radLat2
        b = self.rad(longitude1) - self.rad(longitude2)
        s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
        s = s * self.EARTH_REDIUS
        return s

    def RoadCost(self,start,end):
        return self.Map[start][end]


    def fit(self):
        #转换成4000的ID
        for i in range(len(self.NodeID)):
            self.NodeID[i] = self.NodeIDList.index(self.NodeID[i])

        #给每一个区域分配初始Node
        if self.FocusOnLocalRegion == False:
            self.LocalRegionBound = (104.00767,104.13,30.6119,30.7092)

        #计算
        #------------------------------------------
        IntervalWidth = (self.LocalRegionBound[1] - self.LocalRegionBound[0]) / self.NumGrideWidth
        IntervalHeight = (self.LocalRegionBound[3] - self.LocalRegionBound[2]) / self.NumGrideHeight

        CenterNodeLocationList = []

        NowDown = self.LocalRegionBound[2]
        for i in range(self.NumGrideHeight):
            NextDown = NowDown + IntervalWidth
            HeightCenter = (NextDown + NowDown) / 2

            NowLeft = self.LocalRegionBound[0]
            for j in range(self.NumGrideWidth):
                NextLeft = NowLeft + IntervalWidth
                WidthCenter = (NextLeft + NowLeft) / 2
                CenterNodeLocationList.append((round(HeightCenter,5),round(WidthCenter,5)))
                NowLeft = NextLeft

            NowDown = NextDown
        #------------------------------------------

        #print(CenterNodeLocationList)
        self.NodeLocation = self.NodeLocation.tolist()
        #print(self.NodeLocation)
        
        for i in CenterNodeLocationList:
            ShortestNodeLocationList = []
            ShortestNodeLocationList.append(self.NodeLocation[0])
            ShortestDistanceCost = self.GetDistance(i[1],i[0],self.NodeLocation[0][0],self.NodeLocation[0][1])
            for j in self.NodeLocation:
                TempDistanceCost = self.GetDistance(i[1],i[0],j[0],j[1])
                if TempDistanceCost < ShortestDistanceCost:
                    ShortestNodeLocationList.append(j)
                    ShortestDistanceCost = TempDistanceCost
            #self.ShortestNodeList.append((ShortestNodeLocation,ShortestDistanceCost))
            #队尾最近
            ShortestNodeLocationList.reverse()
            for j in ShortestNodeLocationList:
                SameNodeInShortestNodeListFLAG = True
                for k in self.ShortestNodeList:
                    if self.NodeID[self.NodeLocation.index(j)] in k:
                        SameNodeInShortestNodeListFLAG = False
                        break
                #集合里没有相同的点
                if SameNodeInShortestNodeListFLAG == True:
                    self.ShortestNodeList.append([self.NodeID[self.NodeLocation.index(j)]])
                    break

        #Prepare for TransportationCluster 
        #-----------------------------------------
        Result = []
        for i in self.ShortestNodeList:
            Result.append([i,0])

        UnallocatedNodeSet = self.NodeID[:]
        if not type(UnallocatedNodeSet) == list:
            UnallocatedNodeSet = UnallocatedNodeSet.tolist()

        for i in Result:
            UnallocatedNodeSet.remove(i[0][0])
        #-----------------------------------------
        
        #TransportationCluster
        #-----------------------------------------
        MinClusterSet = []
        while UnallocatedNodeSet:
            #print(len(UnallocatedNodeSet))
            #Find Minimum Cluster
            MinClusterSet.clear()
            MinCluster = Result[0]
            for i in Result:
                if i[1] <= MinCluster[1]:
                    MinClusterSet.append(i)
            MinCluster = random.choice(MinClusterSet)
            #print(MinCluster)
            #End

            #MinCluster = ([2709], 0)
            #Find Shortest Node
            ShortestNode = UnallocatedNodeSet[0]
            ShortestNodeCost = self.RoadCost(MinCluster[0][0],ShortestNode)
            for i in MinCluster[0]:
                for j in UnallocatedNodeSet:
                    if self.RoadCost(i,j) < ShortestNodeCost:
                        ShortestNode = j
                        ShortestNodeCost = self.RoadCost(i,j)
                        if ShortestNodeCost == 0:
                            break
            #End
            #print(ShortestNode,ShortestNodeCost)

            #MinCluster = MinCluster U ShortestNode
            MinCluster[0].append(ShortestNode)
            MinCluster[1] = MinCluster[1] + ShortestNodeCost

            #UnallocatedNodeSet = UnallocatedNodeSet - ShortestNode
            UnallocatedNodeSet.remove(ShortestNode)
        #-----------------------------------------
        #End

        for i in self.NodeID:
            for j in range(len(Result)):
                if i in Result[j][0]:
                    self.labels_.append(j)

        self.labels_ = np.array(self.labels_)

        return


class TransportationCluster2(TransportationCluster):

    def fit(self):
        #转换成4000的ID
        for i in range(len(self.NodeID)):
            self.NodeID[i] = self.NodeIDList.index(self.NodeID[i])

        #给每一个区域分配初始Node
        if self.FocusOnLocalRegion == False:
            self.LocalRegionBound = (104.00767,104.13,30.6119,30.7092)

        #计算
        #------------------------------------------
        IntervalWidth = (self.LocalRegionBound[1] - self.LocalRegionBound[0]) / self.NumGrideWidth
        IntervalHeight = (self.LocalRegionBound[3] - self.LocalRegionBound[2]) / self.NumGrideHeight

        CenterNodeLocationList = []

        NowDown = self.LocalRegionBound[2]
        for i in range(self.NumGrideHeight):
            NextDown = NowDown + IntervalWidth
            HeightCenter = (NextDown + NowDown) / 2

            NowLeft = self.LocalRegionBound[0]
            for j in range(self.NumGrideWidth):
                NextLeft = NowLeft + IntervalWidth
                WidthCenter = (NextLeft + NowLeft) / 2
                CenterNodeLocationList.append((round(HeightCenter,5),round(WidthCenter,5)))
                NowLeft = NextLeft

            NowDown = NextDown
        #------------------------------------------

        #print(CenterNodeLocationList)
        self.NodeLocation = self.NodeLocation.tolist()
        #print(self.NodeLocation)
        
        for i in CenterNodeLocationList:
            ShortestNodeLocationList = []
            ShortestNodeLocationList.append(self.NodeLocation[0])
            ShortestDistanceCost = self.GetDistance(i[1],i[0],self.NodeLocation[0][0],self.NodeLocation[0][1])
            for j in self.NodeLocation:
                TempDistanceCost = self.GetDistance(i[1],i[0],j[0],j[1])
                if TempDistanceCost < ShortestDistanceCost:
                    ShortestNodeLocationList.append(j)
                    ShortestDistanceCost = TempDistanceCost
            #self.ShortestNodeList.append((ShortestNodeLocation,ShortestDistanceCost))
            #队尾最近
            ShortestNodeLocationList.reverse()
            for j in ShortestNodeLocationList:
                SameNodeInShortestNodeListFLAG = True
                for k in self.ShortestNodeList:
                    if self.NodeID[self.NodeLocation.index(j)] in k:
                        SameNodeInShortestNodeListFLAG = False
                        break
                #集合里没有相同的点
                if SameNodeInShortestNodeListFLAG == True:
                    self.ShortestNodeList.append([self.NodeID[self.NodeLocation.index(j)]])
                    break

        #Prepare for TransportationCluster 
        #-----------------------------------------
        Result = []
        for i in self.ShortestNodeList:
            Result.append([i,0])

        UnallocatedNodeSet = self.NodeID[:]
        if not type(UnallocatedNodeSet) == list:
            UnallocatedNodeSet = UnallocatedNodeSet.tolist()

        for i in Result:
            UnallocatedNodeSet.remove(i[0][0])
        #-----------------------------------------
        
        #TransportationCluster
        #-----------------------------------------
        MinClusterSet = []
        while UnallocatedNodeSet:
            #print(len(UnallocatedNodeSet))
            #Find Minimum Cluster
            MinClusterSet.clear()
            MinCluster = Result[0]
            for i in Result:
                if i[1] <= MinCluster[1]:
                    MinClusterSet.append(i)
            MinCluster = random.choice(MinClusterSet)
            #print(MinCluster)
            #End

            #MinCluster = ([2709], 0)
            #Find Shortest Node
            ShortestNode = UnallocatedNodeSet[0]
            ShortestNodeCost = 2 * self.RoadCost(MinCluster[0][0],ShortestNode)
            for i in MinCluster[0]:
                for j in UnallocatedNodeSet:
                    if self.RoadCost(i,j) + self.RoadCost(MinCluster[0][0],j) < ShortestNodeCost:
                        ShortestNode = j
                        ShortestNodeCost = self.RoadCost(i,j) + self.RoadCost(MinCluster[0][0],j)
                        if ShortestNodeCost == 0:
                            break
            #End
            #print(ShortestNode,ShortestNodeCost)
            
            #MinCluster = MinCluster U ShortestNode
            MinCluster[0].append(ShortestNode)
            MinCluster[1] = MinCluster[1] + ShortestNodeCost

            #UnallocatedNodeSet = UnallocatedNodeSet - ShortestNode
            UnallocatedNodeSet.remove(ShortestNode)
        #-----------------------------------------
        #End

        for i in self.NodeID:
            for j in range(len(Result)):
                if i in Result[j][0]:
                    self.labels_.append(j)

        self.labels_ = np.array(self.labels_)

        return
