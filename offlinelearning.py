import os
import random
import math
import re
import pandas as pd
import numpy as np
#import multiprocessing as mp

from dqn.KerasAgent import DQNAgent

from tqdm import tqdm




#读取记忆文件夹得到记忆的位置
def ReadSaveNum(filepath = "./remember/"):
    RememberList = os.listdir(filepath)
    RememberIntList = []
    for i in RememberList:
        temp = re.findall(r'_\d+_', i)
        if len(temp):
            RememberIntList.append(temp[0])
    RememberNewIntList = []
    if len(RememberIntList):
        for i in RememberIntList:
            i = i[1:-1]
            i = int(i)
            RememberNewIntList.append(i)
        RememberNewIntList.sort()
        res = RememberNewIntList[-1] + 1
    else:
        res = 0
    return res



def ReadRemember(ClustersNum=125,VehiclesNum=10000,SaveNum=0):
    input_file_path = "./remember/" + str(ClustersNum) + 'Cluster' + str(VehiclesNum) + 'Vehicles_' + str(SaveNum) + '_Exp.csv'
    reader = pd.read_csv(input_file_path,chunksize = 1000)
    Remember = []
    for chunk in reader:
        Remember.append(chunk)
    Remember = pd.concat(Remember)
    Remember = Remember.values
    return Remember


def str2list(str):
    res = []
    temp = re.findall(r'\d+', str)
    for i in temp:
        res.append(int(i))
    return res



action_size = 13

state_size = 5 + 3 + (action_size) * 2
ClusterStateSize = 3 + (action_size) * 2




DQN = DQNAgent(state_size, action_size,learning_rate = 0.00001)

#DQN.load("./model/ddqn小状态.h5")

MaxSaveNum = ReadSaveNum()


for i in range(MaxSaveNum):
    Remember = ReadRemember(SaveNum = i)

    random.shuffle(Remember)
    print("读取第",i,"个记忆")

    #for j in tqdm(range(len(Remember))):
    for j in range(len(Remember)):
        State = Remember[j][0]
        Action = Remember[j][1]
        Reward = Remember[j][2]
        State_ = Remember[j][3]

        State = str2list(State)
        State = np.reshape(State, [1, DQN.state_size])
        State_ = str2list(State_)
        State_ = np.reshape(State_, [1, DQN.state_size])

        #print(type(State),State)
        #print(type(Action),Action)
        #print(type(Reward),Reward)
        #print(type(State_),State_)

        DQN.remember(State, Action, Reward, State_, False)


        #经验回放
        #------------------------------------------------ 
        if len(DQN.memory) > DQN.batch_size and j % (12*DQN.batch_size) == 0:

            for k in range(16):
                ExpLossHistory = DQN.replay(DQN.batch_size)

            if DQN.epsilon > DQN.epsilon_min:
                DQN.epsilon *= DQN.epsilon_decay

            print("loss均值",round(np.mean(ExpLossHistory),5),"loss方差",round(np.var(ExpLossHistory),5),"epsilon: ",round(DQN.epsilon,5))
        #------------------------------------------------

        #更换参数
        #---------------------------------------------
        #if (step > 30) and (step % 10 == 0):
        if (j % 50000 == 0):
            DQN.update_target_model()
            print("更换参数")
        #---------------------------------------------

    #if i%5 == 0 and i>50:
    if i%5 == 0:
        DQN.save("./model/offline/"+"125Cluster10000Vehicles_ddqn"+str(i)+".h5")


    if i == 15:
        learning_rate = 0.000005



