import numpy as np


def loadExData():
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


# if __name__ == '__main__':
#     Data = loadExData()
#     U, Sigma, VT = np.linalg.svd(Data)
#     print(Sigma)
#     Sig3 = np.mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
#     print(U[:, :3] * Sig3 * VT[:3, :])


def euclidSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# if __name__ == '__main__':
#     myMat = np.mat(loadExData())
#     print(euclidSim(myMat[:, 0], myMat[:, 4]))
#     print(euclidSim(myMat[:, 0], myMat[:, 0]))
#     print(cosSim(myMat[:, 0], myMat[:, 4]))
#     print(cosSim(myMat[:, 0], myMat[:, 0]))
#     print(pearsSim(myMat[:, 0], myMat[:, 4]))
#     print(pearsSim(myMat[:, 0], myMat[:, 0]))


# 示例：餐馆菜肴推荐引擎
def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print(str.format("the {:d} and {:d} similarity is: {:f}", item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything!'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# if __name__ == '__main__':
#     myMat = np.mat(loadExData())
#     # 原代码的矩阵数据有误
#     myMat[4, 1] = myMat[4, 0] = myMat[5, 0] = myMat[6, 0] = 4
#     myMat[0, 3] = 2
#     print(recommend(myMat, 6))
#     print(recommend(myMat, 6, simMeas=euclidSim))
#     print(recommend(myMat, 6, simMeas=pearsSim))


# 示例：基于SVD的图像压缩
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end=' ')
            else:
                print(0, end=' ')
        print()


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("***original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print(str.format("****reconstructed matrix using {:d} singular values******", numSV))
    printMat(reconMat, thresh)


if __name__ == '__main__':
    imgCompress(2)



