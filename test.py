import numpy as np
from perceptron import Perceptron
#creationdesdonneesapprentissage
#entrees:4vecteursd’entreededimension2
X_train=[]
X_train.append(np.array([1,1]))
X_train.append(np.array([1,0]))
X_train.append(np.array([0,1]))
X_train.append(np.array([0,0]))
#sorties:classescorrespondantaux4entrees
y_train=np.array([1,-1,-1,-1])
#creationduclassifieur
perceptron=Perceptron(dimension=2,max_iter=100,learning_rate=0.1)
#entrainementduclassifieur−>calculdescoefficientsdel’hyperplan
perceptron.fit(X_train,y_train)
#predictiondenouvellesentrees
new_x=np.array([1,1])
print(perceptron.predict(new_x))
new_x=np.array([0,1])
print(perceptron.predict(new_x))