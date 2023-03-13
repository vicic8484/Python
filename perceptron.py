import numpy as np


class Perceptron:
    def __init__(self, dimension, max_iter, learning_rate):
        self.dimension = dimension
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = np.zeros(self.dimension)
        self.b = 0

    def fit(self, X, y):
        if(len(X) < self.max_iter):
            for i in range(len(X)):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += self.learning_rate * y[i] * X[i]
                    self.b += self.learning_rate * y[i]
        else:
            for i in range(self.max_iter):
                rand = np.random.randint(0, len(X))
                if y[rand] * (np.dot(self.w, X[rand]) + self.b) <= 0:
                    self.w += self.learning_rate * y[rand] * X[rand]
                    self.b += self.learning_rate * y[rand]
    
    def predict(self, X):
        return np.sign(np.dot(self.w, X) + self.b)
