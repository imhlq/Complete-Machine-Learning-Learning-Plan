# Logistic Regression
# Corresponding to [Prblem 3.3]

import numpy as np
import pandas as pd

def createData(filename, columns, label):
    # waterlemon 3a
    dt = pd.read_csv(filename)
    xi = np.array(dt[columns])
    yi = np.array(dt[label])
    return xi, yi


class LogisticRegression:
    def __init__(self, xi, yi):
        self.N_in = np.shape(xi)[0]
        self.N_dim = np.shape(xi)[1]

        new_col = np.ones((self.N_in, 1))
        self.xi = np.hstack((xi, new_col))

        self.yi = yi


    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))


    def likelihood(self, y_new):
        return np.sum(- self.yi * y_new + np.log(1+np.exp(y_new)))

    def initPara(self):
        self.beta = np.random.rand(self.N_dim + 1, 1) # with bias b
        self.output_l()

    def gd(self, n=10000, learning_rate=0.01):
        print('Begin GD...')
        # calculate partial derivatives
        for i in range(n):
            y_new = np.matmul(self.xi, self.beta)
            p1 = np.exp(y_new) / (1+np.exp(y_new))
            beta_deriv = -np.sum(self.xi * (self.yi-p1), axis=0).reshape(-1, 1)

            self.beta = self.beta - beta_deriv * learning_rate

    def output_l(self):
        y_new = np.matmul(self.xi, self.beta)
        l0 = self.likelihood(y_new)
        print('======')
        print('Para Beta, with l0=%.3f' % l0)
        print(self.beta)
        print('======')

    def generatePredictFunction(self):
        def func(xi, threshold=0.5):
            xi = np.append(xi, 1)
            return np.where(np.matmul(xi, self.beta) > threshold, 1, 0)
        return func

    def train(self):
        self.initPara()
        self.gd()
        self.output_l()

if __name__ == "__main__":
    xi, yi = createData('iris.csv', ['sepal length', 'sepal width', 'petal length', 'petal width'], ['label'])
    LR = LogisticRegression(xi, yi)
    LR.initPara()
    LR.gd()
    LR.output_l()
    input('>Press to show data<')
    # Valid Data
    yfunc = LR.generatePredictFunction()
    correct = 0
    for i in range(len(yi)):
        yi_ = yfunc(xi[i])
        if yi_ == yi[i]:
            correct += 1
        print('Predict:%d, Real:%d' % (yi_, yi[i]))
    print('Total:%d, Correct:%d, Error:%d, Correct_Rate:%.2f%%' % (len(yi), correct, len(yi)- correct, correct/len(yi)*100))