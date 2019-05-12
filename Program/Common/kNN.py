#
# k-Nearset Neighbors algorithm 
# 距离投票法
#
import numpy as np

def createDataSet():
    group = np.array([[1,1.1], [1, 1], [0,0], [0,0.1]])
    labels = ['A', 'A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    # Distance Calculation
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = np.sqrt(sqDistances)
    sortedDistInd = distances.argsort()
    # Vote k
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistInd[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # Dict Sort
    sortedDict = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedDict[0][0]

# verify
da, la = createDataSet()
print(classify0([.1, .1], da, la, 2))

