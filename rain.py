from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
import sys 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

dataset = loadtxt('dataset.csv', delimiter=",")
X = dataset[:,1:4]
y = dataset[:,4]

#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, shuffle = True)
print(X_train.shape, X_test.shape)

#Modelo
model = Sequential()
model.add(Dense(100, input_dim=3, activation='sigmoid'))
model.add(Dense(33, activation='sigmoid'))
model.add(Dense(11, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#Compilacao
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

#Treinamento (fit)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=100)
print(history.history.keys())

#Grafico da acuracia
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Grafico do loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Avaliacao

predict = model.predict_classes(X_test)
cm = confusion_matrix(y_test, predict)
print(cm)

vn, fp, fn, vp = confusion_matrix(y_test, predict).ravel()

v_positivo = vp / (vp + fn)
v_negativo = vn / (vn + fp)
ac = (vp+vn) / (vp+vn+fn+fp)

print("Verdadeiro positivo: ", v_positivo)
print("Verdadeiro negativo: ", v_negativo)
print("Acuracia: ", ac)
