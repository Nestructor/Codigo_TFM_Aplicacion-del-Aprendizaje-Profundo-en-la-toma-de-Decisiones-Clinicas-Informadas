# Clasificación en MIMIC-III
 
 
# - Código 'Clasificacion.py'   
# - Trabajo Fin de Máster.   
# - Néstor Bolaños Bolaños. (nestorbolanos@correo.ugr.es)

# Predicción de la mortalidad intrahospitalaria usando One-Hot Encoding y undersampling
# Vamos a predecor la mortalidad intrahospitalaria en las primeras 48 horas de la estancia en la UCI. Las variables categóricas se han codificado mediante One-Hot Encoding en este cuaderno.

# Instalación de bibliotecas en caso necesario
# !pip3 install sklearn
# !pip3 install seaborn
# !pip3 install tensorflow

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from matplotlib import pyplot as plt
import seaborn as sn
import sys
import plot_metric
import tensorflow as tf
import imp

# Preparación de los datos preprocesados

# Obtenemos los datos de mortalidad intrahospitalaria preprocesados en el cuaderno anterior:
X = pd.read_hdf('datos_preprocesados.h5', 'X')
Y = pd.read_hdf('datos_preprocesados.h5', 'Y')[['in_hospital_mortality', 'los']]

X = X.sort_index(axis = 0, level = 'icustay_id')
Y = Y.sort_index(axis = 0, level = 'icustay_id')

# Eliminamos los datos relacionados con las estancias en la UCI inferiores a 48 horas
indices_a_eliminar = []
for i, fila in Y.iterrows():
    if fila['los'] < 48:
        indices_a_eliminar.append(i)
        
X = X.reset_index().set_index('icustay_id').drop(indices_a_eliminar, axis = 0)
Y = Y.drop(indices_a_eliminar, axis = 0)

"""
Columnas de la matriz:
-Identificador del paciente
-Identificador de la estancia en la UCI
-Identificador de la hora de entrada
-Tiempo en la UCI (en horas)
"""

# Matriz con datos de entrada
def create_x_matrix(x): # obtenemos los datos de las estancias en la UCI de las primeras 48 horas
    zeros = np.zeros((48, x.shape[1]-4))
    x = x.values
    x = x[:48, 4:]
    return (zeros[0:x.shape[0], :] = x)

# Matriz con datos de salida
def create_y_matrix(y):
    return (y['in_hospital_mortality'].to_numpy())


x = np.array(list(X.reset_index().groupby('icustay_id').apply(create_x_matrix)))
y = np.array(list(Y.groupby('icustay_id').apply(create_y_matrix)))[:, 0]


# Métodos de balanceo para el desequilibrio de las clases
 
# Con este método, reduciremos el número de muestras de mortalidad no hospitalaria para igualar el número de muestras de mortalidad hospitalaria.

def undersample_majority(x_train, y_train):
    
    # Separamos los valores x/y, incluyendo las etiquetas:
    muestras_positivas = x_train[y_train == 1]
    muestras_negativas = x_train[y_train == 0]
    etiquetas_positivas = y_train[y_train == 1]
    etiquetas_negativas = y_train[y_train == 0]
    
    # Contamos el número de muestras negativas, y seleccionamos dicha cantidad en las muestras positivos.
    # Con ello, las muestras positivas se copiarán, o repetirán, aparenciendo varias veces en los datos de entrenamiento 
    ids = np.arange(len(muestras_negativas))
    choices = np.random.choice(ids, len(muestras_positivas))
    res_neg_features = muestras_negativas[choices]
    res_neg_labels = etiquetas_negativas[choices]
    
    # Combinamos el remuestreo realizado, y las muestras negativas, mezclándolas
    muestras_remuestreadas = np.concatenate([res_neg_features, muestras_positivas], axis=0)
    etiquetas_remuestreadas = np.concatenate([res_neg_labels, etiquetas_positivas], axis=0)
    order = np.arange(len(etiquetas_remuestreadas))
    np.random.shuffle(order)
    muestras_remuestreadas = muestras_remuestreadas[order]
    etiquetas_remuestreadas = etiquetas_remuestreadas[order]
    return muestras_remuestreadas, etiquetas_remuestreadas


# Regresión Logística
 
# Antes de utilizar el modelo LSTM, utilizaremos el de regresión logística para realizar una comparación.

# Validación cruzada estratificada de 5 pliegues
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
preds_reg_log = []
y_verdaderos_reg_log = []
for indices_train, indices_test in kf.split(x, y):
    
    # Separamos en entrenamiento y test, de forma que la proporción sea 87.5% y 12.5% respectivamente
    x_train, x_val, y_train, y_val = train_test_split(x[train_index], y[train_index], test_size=0.125, random_state=0, 
                                            stratify=y[train_index])
    x_test, y_test = x[indices_test], y[indices_test]
    
    x_train, y_train = undersample_majority(x_train, y_train)
    
    # Ordenamos:
    x_train_reg_log = np.reshape(x_train, (x_train.shape[0], -1))
    x_val_reg_log = np.reshape(x_val, (x_val.shape[0], -1))
    x_test_reg_log = np.reshape(x_test, (x_test.shape[0], -1))
    
    # Aplicamos el modelo
    reg_log = LogisticRegression(penalty = 'l2', C = 1, random_state = 0)
    reg_log.fit(x_train_reg_log, y_train)
    
    # Obtenemos las predicciones
    pred = reg_log.predict_proba(x_test_reg_log)
    preds_reg_log.append(list(pred))
    y_verdaderos_reg_log.append(list(y_test))

# Guardamos los resultados obtenidos:
y_verdaderos_reg_log_combinados = np.concatenate(y_verdaderos_reg_log)
preds_reg_log_combinadas = np.concatenate(preds_reg_log)


# Modelo LSTM

# Preparamos los arrays para guardar los resultados
y_verdaderos_lstm, predicciones_lstm = [], []
loss_train, loss_val = [], []
acc_train, acc_val, auc_train, auc_val = [], [], [], []

# Validación cruzada estratificada de 5 pliegues
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
for indices_train, indices_test in kf.split(x, y):
    
    # Separamos en entrenamiento y test, de forma que la proporción sea 87.5% y 12.5% respectivamente
    x_train, x_val, y_train, y_val = train_test_split(x[indices_train], y[indices_train], test_size=0.125, random_state=0, 
                                            stratify=y[indices_train])
    x_test, y_test = x[indices_test], y[indices_test]
    
    x_train, y_train = undersample_majority(x_train, y_train)
    
    # Inicializamos el modelo con sus parámetros
    modelo = imp.load_source(os.path.basename('lstm.py'), 'lstm.py')
    lstm = modelo.Network(dim=16, batch_norm=False, dropout=0.3, depth=2, rec_dropout=0.0, task='ihm', batch_size=8)
    lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), loss='binary_crossentropy', 
                 loss_weights=None, metrics = ['accuracy', tf.keras.metrics.AUC(name = 'auc')])
    
    # Realizamos el entrenamiento:
    control_modelo = tf.keras.callbacks.ModelCheckpoint(filepath='lstm_one_hot_encoding', monitor='val_auc', 
                                                      verbose=1, mode='max', save_weights_only=True, save_best_only=True)
    historial = lstm.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), epochs=50, batch_size=8, 
                       callbacks=[PlotMetrics(), control_modelo])

    # Obtenemos las predicciones
    lstm.load_weights('lstm_one_hot_encoding')
    pred = lstm.predict(x_test, batch_size=8, verbose=1)
    predicciones_lstm.append(list(pred))
    y_verdaderos_lstm.append(list(y_test))
    
    # Guardamos los resultados incluyendo algunas métricas.
    loss_train.append(historial.historial['Pérdida'])
    loss_val.append(historial.historial['Pérdida Validación'])
    acc_train.append(historial.historial['Precisión'])
    acc_val.append(historial.historial['Precisión Validación'])
    auc_train.append(historial.historial['AUC'])
    auc_val.append(historial.historial['AUC Validación'])

# Estudiamos los resultados de la métrica "loss" en cada pliegue:
fig, axs = plt.subplots(3, 5, figsize=(20, 8))
for i in range(5):
    axs[0, i].plot(np.arange(len(loss_train[i])), loss_train[i], label = 'Entrenamiento')
    axs[0, i].plot(np.arange(len(loss_val[i])), loss_val[i], label = 'Validación')
    axs[0, i].set_title('Pliegue ' + str(i + 1))
    
    axs[1, i].plot(np.arange(len(acc_train[i])), acc_train[i], label = 'Entrenamiento')
    axs[1, i].plot(np.arange(len(acc_val[i])), acc_val[i], label = 'Validación')
    
    axs[2, i].plot(np.arange(len(auc_train[i])), auc_train[i], label = 'Entrenamiento')
    axs[2, i].plot(np.arange(len(auc_val[i])), auc_val[i], label = 'Validación')
    axs[2, i].set_xlabel('Épocas')
    
axs[0, 0].set_ylabel('Entropía Cruzada Binaria')
axs[1, 0].set_ylabel('Precisión')
axs[2, 0].set_ylabel('AUC')
axs[0, 4].legend(loc = 'upper right')
axs[1, 4].legend(loc = 'upper right')
axs[2, 4].legend(loc = 'upper right')


y_verdaderos_lstm_combinados = np.concatenate(y_verdaderos_lstm)
predicciones_lstm_combinadas = np.concatenate(predicciones_lstm)


# Resultados

fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# Matriz de confusión para el modelo LSTM
matriz_confusion = metrics.confusion_matrix(y_verdaderos_lstm_combinados, np.round(predicciones_lstm_combinadas), normalize = 'true')
df_cf = pd.DataFrame(np.round(matriz_confusion, 2), index = ['No', 'Yes'], columns = ['No', 'Yes'])
sn.heatmap(df_cf, annot=True, cmap="viridis", ax = ax[0])
ax[0].set_title('Mortalidad Intrahopitalaria utilizando LSTM \n (Codificación One Hot)')
ax[0].set_xlabel('Etiqueta Obtenida')
ax[0].set_ylabel('Etiqueta Verdadera')

# Matriz de confusión para el modelo de regresión logística
matriz_confusion = metrics.confusion_matrix(y_verdaderos_reg_log_combinados, np.argmax(preds_reg_log_combinadas, axis = 1), normalize = 'true')
df_cf = pd.DataFrame(np.round(matriz_confusion, 2), index = ['No', 'Yes'], columns = ['No', 'Yes'])
sn.heatmap(df_cf, annot=True, cmap="viridis", ax = ax[1])
ax[1].set_title('Mortalidad Intrahopitalaria utilizando RL \n (Codificación One Hot)')
ax[1].set_xlabel('Etiqueta Obtenida')
ax[1].set_ylabel('Etiqueta Verdadera')

# Representamos los resultados mediante el área bajo la curva ROC
roc_auc_reg_log = metrics.roc_auc_score(y_verdaderos_reg_log_combinados, preds_reg_log_combinadas[:, 1])
fpr_reg_log, tpr_reg_log, threshold_reg_log = metrics.roc_curve(y_verdaderos_reg_log_combinados, preds_reg_log_combinadas[:, 1])

roc_auc_lstm = metrics.roc_auc_score(y_verdaderos_lstm_combinados, predicciones_lstm_combinadas)
fpr_lstm, tpr_lstm, threshold_lstm = metrics.roc_curve(y_verdaderos_lstm_combinados, predicciones_lstm_combinadas)

fig, ax = plt.subplots(figsize = (7, 7))
ax.plot(fpr_reg_log, tpr_reg_log, label = 'Regresión Logística (AUC = ' + str(round(roc_auc_reg_log, 2)) + ')')
ax.plot(fpr_lstm, tpr_lstm, label = 'LSTM (AUC = ' + str(round(roc_auc_lstm, 2)) + ')')
ax.plot([0, 1], [0, 1], linestyle='--', color = 'gray', label = 'Chance', alpha=0.8)
plt.legend(loc='lower right')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Área bajo la curva ROC para la predicción de mortalidad intrahopitalaria \n (Codificación One Hot)')