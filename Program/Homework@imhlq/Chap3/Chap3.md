# Chap3 Homework

## 1. Bias b

When input=0, output=0

## 2. Logistic function and likelihood function

Logistic function: 2-order derivative:
~ $1-\exp(-(wx+b))<>0$ ~ $wx+b<>0$

Likelihood function: 2-order derivative:
~ $x^2 \exp(wx+b) > 0$

## 3. Logistic Regression On waterlemon 3$\alpha$

```
# Logistic Regression
# Corresponding to [Prblem 3.3]

import numpy as np
import pandas as pd

def createData():
    # waterlemon 3a
    dt = pd.read_csv('watermelon3_0_En.csv')
    xi = np.array(dt[['Density', 'SugerRatio']])
    yi = np.array(dt[['Label']])
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

    def gd(self, n=20000, learning_rate=0.01):
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

if __name__ == "__main__":
    xi, yi = createData()
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
```

Result:
```
Beta:
[[ 3.11909998]
 [12.37100294]
 [-4.37694282]]

l:
8.684

Total:17, Correct:12, Error:5, Correct_Rate:70.59%
```

## 4. 10-fold and keep-one

Just have tried, it will work



## 5. LDA

```
python LDA.py

result:
w=
 [[  1.42324415]
 [  9.20684517]
 [-10.61587897]
 [-16.29492943]]
Total:100, Correct:100, Error:0, Correct_Rate:100.00%
```

