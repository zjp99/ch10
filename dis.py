
# -*- coding: utf-8 -*-#

"""
File         :      dis.py
Description  :  
Author       :      赵金朋
Modify Time  :      2019/6/20 17:41
"""
# -*- coding: utf-8 -*-#

"""
File         :      ccc.py
Description  :
Author       :      赵金朋
Modify Time  :      2019/5/5 11:31
"""
from numpy import *



# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def kMeans(dataSet,  distMeas=distEclud ):
    centroids=loadDataSet('places.txt')
    m = 150
    clusterAssment = mat(zeros((m, 2)))  # 与80*2矩阵，第一列存取距离最小的索引，第二列存储距离
    for i in range(m):
        minDist = inf
        minIndex = -1
#循环质心个数的次数
        for j in range(3):
            distJI = distMeas(centroids[j, :], dataSet[i, :])
            if distJI < minDist:
                minDist = distJI
                minIndex = j
        clusterAssment[i, :] = minIndex, minDist ** 2
    return clusterAssment





if __name__ == '__main__':
    dataSet=loadDataSet('lris.txt')
    #center=loadDataSet('places.txt')
    c=kMeans(dataSet)