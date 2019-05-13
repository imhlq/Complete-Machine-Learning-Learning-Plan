# Linear Discriminant Analysis
# Corresponding to 3.5
import numpy as np
import pandas as pd

class LDAclassify:
    def __init__(self, xi, yi):
        self.n_dim = np.shape(xi)[1]
        # Devide into two parts
        X0 = []
        X1 = []
        for i in range(len(yi)):
            if yi[i] == 0:
                X0.append(xi[i])
            else:
                X1.append(xi[i])
        self.X0 = np.array(X0)
        self.X1 = np.array(X1)

        # Average vector
        self.mu0 = self.X0.mean(axis=0, keepdims=True)
        self.mu1 = self.X1.mean(axis=0, keepdims=True)

        # Covarient Matrix
        self.cov0 = np.cov(self.X0.T)
        self.cov1 = np.cov(self.X1.T)

    def initPara(self):
        self.w = np.matmul(np.linalg.inv(self.Sw), (self.mu0 - self.mu1).T)

    @property
    def Sw(self):
        # within-class scatter matrix
        return self.cov0 + self.cov1

    @property
    def Sb(self):
        # between-class scatter matrix
        Sb = self.mu0 - self.mu1
        Sb = np.matmul(Sb.T, Sb)
        return Sb

    def project(self, x):
        return np.matmul(x, self.w)

    def predict(self, x):
        pmu0 = self.project(self.mu0)
        pmu1 = self.project(self.mu1)
        px = self.project(x)
        return np.where(abs(px - pmu0) > abs(px - pmu1), 1, 0)


    def output(self):
        print('We have: w=\n', self.w)


def createData(filename, columns, label):
    dt = pd.read_csv(filename)
    xi = np.array(dt[columns])
    yi = np.array(dt[label])
    return xi, yi


if __name__ == "__main__":
    xi, yi = createData('C://Users//im_hl//OneDrive - North Carolina State University//Project//CMLLP//Program//Homework@imhlq//Chap3//iris.csv', ['sepal length', 'sepal width', 'petal length', 'petal width'], ['label'])
    lda = LDAclassify(xi, yi)
    lda.initPara()
    lda.output()

    correct = 0
    for i in range(len(yi)):
        yi_ = lda.predict(xi[i])
        if yi_ == yi[i]:
            correct += 1
        print('Predict:%d, Real:%d' % (yi_, yi[i]))
    print('Total:%d, Correct:%d, Error:%d, Correct_Rate:%.2f%%' % (len(yi), correct, len(yi)- correct, correct/len(yi)*100))