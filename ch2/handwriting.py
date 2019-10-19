from kNN import *
import os
import numpy as np


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


# if __name__ == '__main__':
#     testVector = img2vector("digits/testDigits/0_13.txt")
#     print(testVector[0, 0:31])
#     print(testVector[0, 32:63])


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir("digits/trainingDigits")
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = \
            img2vector(str.format("digits/trainingDigits/{:s}", fileNameStr))
    testFileList = os.listdir("digits/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector(str.format("digits/testDigits/{:s}", fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print(str.format("the classifier came back with: {:d} \
                         the real answer is: {:d}", classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print(str.format("the total number of errors is: {:d}", int(errorCount)))
    print(str.format("the total error rate is: {:f}", errorCount / float(mTest)))


if __name__ == '__main__':
    handwritingClassTest()
