def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck: list, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


# if __name__ == '__main__':
#     dataSet = loadDataSet()
#     C1 = createC1(dataSet)
#     print(list(C1))
#     D = list(map(set, dataSet))
#     print(list(D))
#     # 使用集合是为了方便issubset操作
#     L1, suppData0 = scanD(D, C1, 0.5)
#     print(L1)


def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# if __name__ == '__main__':
#     dataSet = loadDataSet()
#     L, supportData = apriori(dataSet)
#     for item in L:
#         print(item)
#     L, supportData = apriori(dataSet, minSupport=0.7)
#     print(L)
#     print(supportData)


def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > m + 1:
        Hmpl = aprioriGen(H, m + 1)
        Hmpl = calcConf(freqSet, Hmpl, supportData, brl, minConf)
        if len(Hmpl) > 1:
            rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)


# if __name__ == '__main__':
#     L, supportData = apriori(loadDataSet(), minSupport=0.5)
#     rules = generateRules(L, supportData, minConf=0.7)
#     print(rules)
#     rules = generateRules(L, supportData, minConf=0.5)
#     print(rules)


# 示例：发现国会投票中的模式
# 需要申请api key


# 示例：发现毒蘑菇的相似特征
if __name__ == '__main__':
    mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, supportData = apriori(mushDataSet, minSupport=0.3)
    for item in L[1]:
        if item.intersection('2'):
            print(item)
    for item in L[3]:
        if item.intersection('2'):
            print(item)
