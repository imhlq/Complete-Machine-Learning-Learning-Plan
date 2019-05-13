# Valid methods
import numpy as np
import matplotlib.pyplot as plt

def method_sample(x):
    return np.clip(np.sum(x, axis=1), 0.0, 1.0)


class Model_Valid:
    def __init__(self, func):
        self.func = func
    
    def formatData(self, dataset):
        xi = dataset[:, :-1]
        yi = dataset[:, -1]
        return xi, yi

    def predict(self, xi, Threshold=0.5):
        return np.where(self.func(xi) >= Threshold, 1, 0)

    def calcValMat(self, xi, yi, Threshold):
        pi = self.predict(xi, Threshold=Threshold)
        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(len(xi)):
            if pi[i] == 1 and yi[i] == 1:
                TP += 1
            if pi[i] == 0 and yi[i] == 1:
                FN += 1
            if pi[i] == 1 and yi[i] == 0:
                FP += 1
            if pi[i] == 0 and yi[i] == 0:
                TN += 1
        return TP, FN, FP, TN


    def calcFscore(self, xi, yi, threshold, beta=1):
        valMat = self.calcValMat(xi, yi, threshold)
        if (valMat[0] + valMat[2]) == 0:
            P = 0
        else:
            P = valMat[0] / (valMat[0] + valMat[2])
        if (valMat[0] + valMat[1]) == 0:
            R = 0
        else:
            R = valMat[0] / (valMat[0] + valMat[1])
        beta2 = beta*beta
        return (1+beta2)*P*R / (beta2 * P + R)
        

    def calcPR(self, xi, yi):
        py = self.func(xi)
        # Sort
        pySortIndex = np.argsort(py)[::-1]
        yiSort = yi[pySortIndex]
        xiSort = xi[pySortIndex]
        pySort = py[pySortIndex]
        # Plot
        xpoint = []
        ypoint = []
        threshold = 1.0
        for i in range(len(pySort)):
            threshold = pySort[i]
            valMat = self.calcValMat(xiSort, yiSort, threshold)
            if (valMat[0] + valMat[2]) == 0:
                P = 0
            else:
                P = valMat[0] / (valMat[0] + valMat[2])
            if (valMat[0] + valMat[1]) == 0:
                R = 0
            else:
                R = valMat[0] / (valMat[0] + valMat[1])
            xpoint.append(R)
            ypoint.append(P)
        return xpoint, ypoint

    def calcROC(self, xi, yi):
        py = self.func(xi)
        # Sort
        pySortIndex = np.argsort(py)[::-1]
        yiSort = yi[pySortIndex]
        xiSort = xi[pySortIndex]
        pySort = py[pySortIndex]
        # Plot
        xpoint = []
        ypoint = []
        threshold = 1.0
        for i in range(len(pySort)):
            threshold = pySort[i]
            valMat = self.calcValMat(xiSort, yiSort, threshold)
            if (valMat[0] + valMat[1]) == 0:
                TPR = 0
            else:
                TPR = valMat[0] / (valMat[0] + valMat[1])
            if (valMat[2] + valMat[3]) == 0:
                FPR = 0
            else:
                FPR = valMat[2] / (valMat[2] + valMat[3])
            xpoint.append(FPR)
            ypoint.append(TPR)
        return xpoint, ypoint

class Learning_Valid:
    def __init__(self, ML):
        # Machine Learing Method Port
        # should include ML.xi, ML.yi, ML.init, ML.train(xi)
        self.ML = ML
    
    def fold_valid(self, n=10):
        xi = self.ML.xi
        yi = self.ML.yi



if __name__ == "__main__":
    mv = Model_Valid(method_sample)
    dataset = np.array([[0, 1], [0.1, 0], [1, 1], [0.9, 1], [0.6, 1], [0.6, 0]])
    roc = mv.calcROC(*mv.formatData(dataset))
    pr = mv.calcPR(*mv.formatData(dataset))
    plt.plot(*roc)
    plt.plot(*pr)
    plt.show()