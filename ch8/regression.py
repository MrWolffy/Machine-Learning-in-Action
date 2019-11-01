import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import json
from urllib import request


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


# if __name__ == '__main__':
#     xArr, yArr = loadDataSet('ex0.txt')
#     ws = standRegres(xArr, yArr)
#     print(ws)
#     xMat = np.mat(xArr)
#     yMat = np.mat(yArr)
#     yHat = xMat * ws
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
#     xCopy = xMat.copy()
#     xCopy.sort(0)
#     yHat = xCopy * ws
#     ax.plot(xCopy[:, 1], yHat)
#     plt.show()
#     yHat = xMat * ws
#     print(np.corrcoef(yHat.T, yMat))


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This martix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


# if __name__ == '__main__':
#     xArr, yArr = loadDataSet('ex0.txt')
#     print(lwlr(xArr[0], xArr, yArr, 1.0))
#     print(lwlr(xArr[0], xArr, yArr, 0.001))
#     yHat = lwlrTest(xArr, xArr, yArr, 0.003)
#     xMat = np.mat(xArr)
#     srtInd = xMat[:, 1].argsort(0)
#     xSort = xMat[srtInd][:, 0, :]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(xSort[:, 1], yHat[srtInd])
#     ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
#     plt.show()


# 示例：预测鲍鱼的年龄
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


# if __name__ == '__main__':
#     abX, abY = loadDataSet('abalone.txt')
#     yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
#     yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
#     yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
#     # 预测误差大小
#     print(rssError(abY[0:99], yHat01.T))
#     print(rssError(abY[0:99], yHat1.T))
#     print(rssError(abY[0:99], yHat10.T))
#     # 新数据集
#     yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
#     print(rssError(abY[100:199], yHat01.T))
#     yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
#     print(rssError(abY[100:199], yHat1.T))
#     yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
#     print(rssError(abY[100:199], yHat10.T))
#     # 简单线性回归
#     ws = standRegres(abX[0:99], abY[0:99])
#     yHat = np.mat(abX[100:199]) * ws
#     print(rssError(abY[100:199], yHat.T.A))


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


# if __name__ == '__main__':
#     abX, abY = loadDataSet('abalone.txt')
#     ridgeWeights = ridgeTest(abX, abY)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(ridgeWeights)
#     plt.show()


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)   # 修改，没有regularize这个函数
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    m, n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# if __name__ == '__main__':
#     xArr, yArr = loadDataSet('abalone.txt')
#     # stageWise(xArr, yArr, 0.01, 200)
#     stageWise(xArr, yArr, 0.001, 5000)


# 示例：预测乐高玩具套装的价格
# 还要申请api就懒得搞了
