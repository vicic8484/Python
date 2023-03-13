import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time
from pickle import dump, load

df=pd.read_csv("iris.csv")
X_data=df.iloc[0:,[0,2]].values
y_data=df.iloc[0:,4].values
y_data=np.where(y_data=="setosa",-1,1)
X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)


perceptron = Perceptron(max_iter=9, tol=0.1)
logistic = LogisticRegression(max_iter=30)
knn = KNeighborsClassifier(n_neighbors=1)

timer = time.time()
perceptron.fit(X_data_train, y_data_train)
print("Perceptron fit: ", (time.time() - timer)*1000)
timer = time.time()
logistic.fit(X_data_train, y_data_train)
print("Logistic fit: ", (time.time() - timer)*1000)
timer = time.time()
knn.fit(X_data_train, y_data_train)
print("KNN fit: ", (time.time() - timer)*1000)


timer = time.time()
print(cross_val_score(perceptron, X_data_train, y_data_train, cv=5))
print("Perceptron crossval: ", (time.time() - timer)*1000)
timer = time.time()
print(cross_val_score(logistic, X_data_train, y_data_train, cv=5))
print("Logistic crossval: ", (time.time() - timer)*1000)
timer = time.time()
print(cross_val_score(knn, X_data_train, y_data_train, cv=5))
print("KNN: crossval", (time.time() - timer)*1000)

dump(perceptron, open('perceptron.pkl', 'wb'))