# -*- coding: utf-8 -*-#

"""
File         :      ccc.py
Description  :
Author       :      赵金朋
Modify Time  :      2019/5/5 11:31
"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import time


# 加载数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 计算向量距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 为给定数据集构建一个包含k个随机质心的集合
# 生成k个随机点
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 列数
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


# 数据集，簇的个数，距离，随机生成质心
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # 与80*2矩阵，第一列存取距离最小的索引，第二列存储距离
    centroids = createCent(dataSet, k)  # 四个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 循环80次，计算最小距离，寻找质心
        for i in range(m):
            minDist = inf
            minIndex = -1
            # 循环质心个数的次数
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        # 更新质心位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 绘制数据集和质心图
def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape
    if dim != 2:
        print("对不起，仅支持二列的数据集！")
        return 1
    # 分类不超过10个
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > 10:
        print("分类过多")
        return 1
    # 绘制数据集
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    # 形状+颜色o为圆，D为方
    mark = ['or', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


# 二分k-均值
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  # 数据集个数
    clusterAssment = mat(zeros((m, 2)))  # m*2矩阵，（所属的中心编号，距中心的距离）
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 对列求均值，然后转存列表 [1*2]
    centList = [centroid0]  # 存储聚类中心[[1*2]]
    print("centList",centList)
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    # 聚类数目小于k
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # 得到dataSet中行号与clusterAssment中所属的中心编号为i的行号对应的子集数据。
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 在给定的簇上进行K-均值聚类,k值为2,centroidMat存取质心, splitClustAss所属质心和距离
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算将该簇划分成两个簇后总误差
            sseSplit = sum(splitClustAss[:, 1])#距离求和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("总误差, and notSplit: ", sseSplit, sseNotSplit)
            # 选择使得误差最小的那个簇进行划分
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将需要分割的聚类中心下的点进行1划分
        # 新增的聚类中心编号为len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssment


import urllib
import json
#从Yahoo!返回一个字典
def geoGrab(stAddress, city):
    apiStem = 'http://api.map.baidu.com/geocoder/v2/?'
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params
    print(yahooApi)
    c = urllib.request.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep
#将所有这些封装起来并且将相关信息保存到文件中
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


"""
#kmean测试
if __name__ == '__main__':
    datMat=mat(loadDataSet('testSet.txt'))
    k=3
    centroids,clusterAssment=kMeans(datMat,k)
    showCluster(datMat,k, centroids, clusterAssment)
"""

#测试bitKmeans
if __name__ == '__main__':
    datMat = mat(loadDataSet('lris.txt'))
    k = 3
    centroids, clusterAssment = biKmeans(datMat, k)
    print('簇中心',centroids)
    showCluster(datMat,k, mat(centroids), clusterAssment)

"""
if __name__ == '__main__':
    clusterClubs(5)
 """
