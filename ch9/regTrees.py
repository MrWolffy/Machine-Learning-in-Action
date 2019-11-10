import numpy as np
import tkinter as tk


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))     # 修改，原代码有bug
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value), :][0]   # 修改，原代码有bug
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value), :][0]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# if __name__ == '__main__':
#     testMat = np.mat(np.eye(4))
#     mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
#     print(mat0)
#     print(mat1)


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n - 1):
        # 原来的代码有bug
        for splitVal in set(dataSet[:, featIndex].flatten().tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


# if __name__ == '__main__':
#     myDat = loadDataSet('ex00.txt')
#     myMat = np.mat(myDat)
#     print(createTree(myMat, regLeaf, regErr))


# if __name__ == '__main__':
#     myDat1 = loadDataSet('ex0.txt')
#     myMat1 = np.mat(myDat1)
#     print(createTree(myMat1, regLeaf, regErr))


# if __name__ == '__main__':
#     myDat2 = loadDataSet('ex2.txt')
#     myMat2 = np.mat(myDat2)
#     # print(createTree(myMat2, regLeaf, regErr))
#     print(createTree(myMat2, regLeaf, regErr, ops=(10000, 4)))


def isTree(obj):
    return type(obj).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
            np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


# if __name__ == '__main__':
#     myDat2 = loadDataSet('ex2.txt')
#     myMat2 = np.mat(myDat2)
#     myTree = createTree(myMat2, regLeaf, regErr, ops=(0, 1))
#     myDatTest = loadDataSet('ex2test.txt')
#     myMat2Test = np.mat(myDatTest)
#     print(prune(myTree, myMat2Test))


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n'
                        'try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


# if __name__ == '__main__':
#     myMat2 = np.mat(loadDataSet('exp2.txt'))
#     print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))


# 示例：树回归与标准回归的比较
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n+1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


# if __name__ == '__main__':
#     trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
#     testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
#     # 回归树
#     myTree = createTree(trainMat, ops=(1, 20))
#     yHat = createForeCast(myTree, testMat[:, 0])
#     print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
#     # 模型树
#     myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
#     yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
#     print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
#     # 标准线性回归
#     ws, X, Y = linearSolve(trainMat)
#     for i in range(np.shape(testMat)[0]):
#         yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
#     print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])


if __name__ == '__main__':
    root = tk.Tk()
    myLabel = tk.Label(root, text="Hello World")
    myLabel.grid()