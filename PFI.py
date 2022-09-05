# Importancia de la característica de permutación (PFI) para la clasificación de latidos utilizando un perceptrón multicapa (MLP)
# 
# 
# - Código 'PFI.py'   
# - Trabajo Fin de Máster.   
# - Néstor Bolaños Bolaños. (nestorbolanos@correo.ugr.es)

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy import *
import os
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import *
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
sns.set()


# Cargamos los datos y codificamos las clases de cada latido mediante One Hot

# Cargamos los datos de entrenamiento y de test:
tamaño = 277
valores_train = np.empty(shape=[0, tamaño])
valores_test = np.empty(shape=[0, tamaño])

latidos_entrenamiento = glob.glob('train_beats.csv')
latidos_test = glob.glob('test_beats.csv')

for j in latidos_entrenamiento:
    filas = np.loadtxt(j, delimiter=',')
    valores_train = np.append(valores_train, filas, axis=0)

for j in latidos_test:
    filas = np.loadtxt(j, delimiter=',')
    valores_test = np.append(valores_test, filas, axis=0)
    
print(valores_train.shape)
print(valores_test.shape)

# Separamos los datos de entrenamiento y de test, y aplicamos la codificación One Hot a Y:
X_train = valores_train[:,:-2]
X_test = valores_test[:,:-2]
y_train = valores_train[:,-2]
y_test = valores_test[:,-2]

# Combinamos todo nuevamente:
X = np.concatenate((X_train, X_test), axis = 0)
Y = np.concatenate((y_train, y_test), axis = 0)

# codificación One Hot de Y:
Y = to_categorical(Y)


# Construimos el perceptrón multicapa

# Construimos el modelo MLP
def getModel():
    model_mlp = Sequential()
    model_mlp.add(Dense(100, activation = 'relu'))
    model_mlp.add(Dense(9, activation = 'softmax'))
    return model_mlp

model_mlp.summary()


# Implementamos y aplicamos PFI para el perceptrón multicapa

# Métodos de perturbación:
# Hay diferentes tipos de perturbación para la importancia de la característica de permutación, como la perturbación media, la perturbación cero y la perturbación aleatoria. En la implementación que hemos realizado en este cuaderno, los datos dentro de cada corte se han barajado aleatoriamente. 

fig, ax = plt.subplots(1, 4, figsize = (20, 4), sharex = True, sharey=True)

# Sin perturbación: señal original.
ax[0].set_title('Sin perturbación')
ax[0].plot(np.arange(len( X[20, :])), X[20, :])

# perturbación 0: se establecen los valores de cada corte a 0.
ax[1].set_title('Perturbación 0')
X_zero_perturbed = X[20, :].copy()
X_zero_perturbed[5 * 25 : 6 * 25] = 0.0
ax[1].plot(np.arange(len(X[20, :])), X_zero_perturbed)

# Perturbación aleatoria: los valores de cada corte se reemplazan con valores aleatorios.
ax[2].set_title('Perturbación aleatoria')
X_random_perturbed = X[20, :].copy()
X_random_perturbed[5 * 25 : 6 * 25] = np.std(X[20, :]) * np.random.randn(25) + np.mean(X[20, :])
ax[2].plot(np.arange(len(X[20, :])), X_random_perturbed)

# Perturbación media: se promedian los valores del corte actual.
ax[3].set_title('Perturbación Media')
X_mean_perturbed = X[20, :].copy()
X_mean_perturbed[5 * 25 : 6 * 25] = np.mean(X[20, 5 * 25 : 6 * 25])
ax[3].plot(np.arange(len(X[20, :])), X_mean_perturbed)

for i in range(4):
    ax[i].set_xlabel('Tiempo')
    ax[i].axvspan(5 * 25, 6 * 25, color = 'green', alpha = 0.25)

# Importancia de la característica de permutación:
kf = StratifiedKFold(n_splits = 5, shuffle = True)
contador_pliegues = 0
M = np.zeros((X.shape[0], 11))
for indice_train, indice_test in kf.split(X, np.argmax(Y, axis = 1)):
    print('Fold ', contador_pliegues)
    
    # Separamos los datos en cada pliegue:
    X_train, X_test = X[indice_train], X[indice_test]
    y_train, y_test = Y[indice_train], Y[indice_test]
    
    # Construimos el modelo de aprendizaje con los datos de entrenamiento:
    model_mlp = getModel()
    model_mlp.compile(optimizer = 'adam', loss = tf.keras.losses.CategoricalCrossentropy())
    model_mlp.fit(X_train, y_train, epochs = 100, verbose = 0)
    
    # Realizamos predicciones con los datos de test sin permutaciones:
    predicciones = model_mlp.predict(X_test)
    
    # Para cada característica:
    for corte in range(0, 275, 25):
        # Permutamos y realizamos predicciones:
        x_permutacion = np.copy(X_test)
        x_corte = X_test[:, corte:corte+25]
        x_corte_permutacion = np.random.permutation(x_corte)
        x_permutacion[:, corte:corte + 25] = x_corte_permutacion
        pred_perm = model_mlp.predict(x_permutacion)
        
        # Obtenemos la importancia:
        importancia = ((np.argmax(y_test, axis = 1) - np.argmax(pred_perm, axis = 1))**2 
                      - (np.argmax(y_test, axis = 1) - np.argmax(predicciones, axis = 1))**2)
        M[indice_test, corte // 25] = importancia 
    contador_pliegues += 1
importancia_media = np.mean(M, axis = 0)

indices_ordenados = np.argsort(-1 * importancia_media)
cortes = np.arange(1, 12)
colores = ['forestgreen', 'limegreen', 'royalblue', 'blue', 'darkorange', 'cyan', 'purple', 'red', 'pink', 'yellow', 'coral']

fig, ax = plt.subplots(1, 2, figsize = (15, 4))
ax[0].bar(range(11), importancia_media[indices_ordenados], color = np.array(colores)[indices_ordenados])
ax[0].set_title('Importancia de cada característica del modelo MLP')
ax[0].set_xticks(np.arange(11))
ax[0].set_xticklabels(cortes[indices_ordenados].astype(int))
ax[0].set_xlabel('Corte')
ax[0].set_ylabel('Importancia de cada característica')

ecg_normalizado = (X[20, :] - X[20, :].min()) / (X[20, :].max() - X[20, :].min())
Importancia_caraceristica_normalizada = (importancia_media - importancia_media.min()) / (importancia_media.max() - importancia_media.min())
ax[1].plot(np.arange(len(ecg_normalizado)), ecg_normalizado, label='Datos ECG')
ax[1].plot(np.repeat(Importancia_caraceristica_normalizada, 25), label = 'Importancia de cada característica')
ax[1].set_title('Importancia de cada característica \npara el modelo MLP en una muestra de ECG')
ax[1].set_xlabel('Tiempo')
ax[1].set_ylabel('Señal ECG / Importancia de cada característica')
ax[1].legend()
