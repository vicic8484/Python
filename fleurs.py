import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

df=pd.read_csv("iris.csv")
#recuperationdes100premiereslignesetdescolonnes0et2
#dutableaucorrespondantauxcaracteristiquessepal_lengthet
#petal_length(entrees)
X_data=df.iloc[0:100,[0,2]].values
#recuperationdes100premiereslignesdeladernierecolonne
#correspondantal’especedel’iris(sorties/classes)
y_data=df.iloc[0:100,4].values
#classificationbinaire:transformationdeslabelsenvaleurs−1ou1
#setosa<=>−1ounonsetosa<=>1
y_data=np.where(y_data=="setosa",-1,1)
#attributiondecouleursdifferentesaux2classes
colors={-1:'red',1:'blue'}
y_colors=[colors[y]for y in y_data]
#affichagedesnuagesdepointsavecleurclasse
#plt.scatter(X_data[:,0],X_data[:,1],c=y_colors,s=100)
#plt.show()

X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)


perceptron = Perceptron(dimension=2, max_iter=10, learning_rate=0.1)
perceptron.fit(X_data_train, y_data_train)

errors = 0
y_pred = []
for i in range(len(X_data_test)):
    y_pred.append(perceptron.predict(X_data_test[i]))
    if  y_pred[i] != y_data_test[i]:
        errors += 1

print("Number of errors: ", errors)
print(perceptron.w)
print(perceptron.b)

print("accuracy score " + str(accuracy_score(y_data_test, y_pred)))
print("confusion matrix \n" + str(confusion_matrix(y_data_test, y_pred)))
print("precision score " + str(precision_score(y_data_test, y_pred)))
print("recall score " + str(recall_score(y_data_test, y_pred)))
print("f1 score " + str(f1_score(y_data_test, y_pred)))
