# -*- coding: utf-8 -*-  
import os
import random
import re
import pandas as pd
import numpy as np
from preprocessing.readfiles import *
from tqdm import tqdm


class CreateAccurateMap(object):

    def __init__(self):
        self.Map = ReadMap('./data/Map__.csv')
        self.Path = ReadPath('./data/Path.csv')
        self.Dimension = len(self.Map)
        self.AccurateMap = np.zeros((self.Dimension,self.Dimension),dtype=np.float32)

    def GetPath(self,i,j,res):
        if (i==j):
             return
        if (self.Path[i][j]==0) : 
            res.append(j)
        else:
            self.GetPath(i,self.Path[i][j],res)
            self.GetPath(self.Path[i][j],j,res)

    def TravelCost(self,path,c='m'):
        i = 0
        j = 1
        dis = 0
        while j<= len(path) - 1 :
            d = self.Map[path[i]][path[j]]
            dis += d
            i += 1
            j += 1
        if c == "d":
            return dis
        elif c == "m":
            res=dis/250 #(250m/min)
            res=round(res,4) #保留3位小数来使时间数据不是<M8
            return res

    def main(self):
        for i in tqdm(range(self.Dimension)):
            for j in range(self.Dimension):
                res = []
                res.append(i)
                self.GetPath(i,j,res)
                self.AccurateMap[i][j] = self.TravelCost(res)

        self.AccurateMap = pd.DataFrame(self.AccurateMap)
        SaveAccurateMapPath = './data/AccurateMap.csv'
        self.AccurateMap.to_csv(SaveAccurateMapPath,header=0,index=0) #不保存列名

if __name__ == '__main__':
    test = CreateAccurateMap()
    test.main()

