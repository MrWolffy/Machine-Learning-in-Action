from math import log
import operator
import treePlotter


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# if __name__ == '__main__':
#     myDat, labels = createDataSet()
#     print(calcShannonEnt(myDat))
#     myDat[0][-1] = 'maybe'
#     print(calcShannonEnt(myDat))


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# if __name__ == '__main__':
#     myDat, labels = createDataSet()
#     print(chooseBestFeatureToSplit(myDat))
#     print(myDat)


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatlabel = labels[bestFeat]
    myTree = {bestFeatlabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatlabel][value] = \
            createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# if __name__ == '__main__':
#     myDat, labels = createDataSet()
#     myTree = createTree(myDat, labels)
#     print(myTree)


# if __name__ == '__main__':
#     treePlotter.createPlot()


# if __name__ == '__main__':
#     myTree = treePlotter.retrieveTree(0)
#     print(treePlotter.getNumLeafs(myTree))
#     print(treePlotter.getTreeDepth(myTree))


# if __name__ == '__main__':
#     myTree = treePlotter.retrieveTree(0)
#     treePlotter.createPlot(myTree)
#     myTree['no surfacing'][3] = 'maybe'
#     treePlotter.createPlot(myTree)


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# if __name__ == '__main__':
#     myDat, labels = createDataSet()
#     myTree = treePlotter.retrieveTree(0)
#     print(classify(myTree, labels, [1, 0]))
#     print(classify(myTree, labels, [1, 1]))


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')   # 此处有修改
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')   # 此处有修改
    return pickle.load(fr)


if __name__ == '__main__':
    myTree = treePlotter.retrieveTree(0)
    storeTree(myTree, 'classifierStorage.txt')
    print(grabTree('classifierStorage.txt'))

