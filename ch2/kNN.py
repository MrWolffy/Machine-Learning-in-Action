import numpy as np
from matplotlib import pyplot as plt
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              # python3已经废除了iteritems方法，改为items
                              # 其作用是返回一个dict的迭代器
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# if __name__ == '__main__':
#     group, labels = createDataSet()
#    print(classify0([0, 0], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# if __name__ == '__main__':
#     datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#     print(datingDataMat)
#     print(datingLabels[0:20])
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
#     #            15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
#     ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#                15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
#     plt.show()


def autoNorm(dataSet: np.ndarray):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# if __name__ == '__main__':
#     datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     print(normMat)
#     print(ranges)
#     print(minVals)



