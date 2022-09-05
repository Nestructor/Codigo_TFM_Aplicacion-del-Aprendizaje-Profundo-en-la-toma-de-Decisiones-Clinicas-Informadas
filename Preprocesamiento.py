# Preprocesamiento en MIMIC-III
 
 
# - Código 'Preprocesamiento.py'   
# - Trabajo Fin de Máster.   
# - Néstor Bolaños Bolaños. (nestorbolanos@correo.ugr.es)

# Preprocesamiento
 
# En el fichero anterior (Extraccion_Datos.ipynb), extraímos 17 variables. En esta parte realizaremos el preprocesamiento antes de comenzar con los modelos de aprendizaje profundo.

# Bibliotecas

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import json


# Lectura de los datos extraídos

eventos = pd.read_hdf('datos_extraidos.h5', 'X')
eventos = eventos.reset_index()

datos_pacientes = pd.read_hdf('datos_extraidos.h5', 'datos_pacientes')

datos = pd.read_hdf('datos_extraidos.h5', 'Y')

# Cargamos un fichero que contiene información sobre variables continuas y categóricas
configuracion = json.load(open('discretizer_config.json', 'r'))
es_categorica = configuracion['is_categorical_channel']

# Obtenemos las variables categóricas
variable_categorica = []
variable_continua = []
for k, valor in es_categorica.items():
    if valor:
        variable_categorica.append(k)
    else:
        variable_continua.append(k)

variable_categorica = variable_categorica[1:]


# Conversión de unidades

# Los datos que provienen de pruebas de laboratorio o de signos vitales, pueden tener diferentes unidades de medida tomadas en la unidad de cuidados intensivos. Por lo tanto, necesitamos una forma de estandarizar las mediciones tomadas y asegurarnos de que se expresan en unidades consistentes.


# Conversión de unidades
CONVERSION_UNIDADES = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),
    ('height',                   'in',  None,             lambda x: x*2.54),
]

nombre_variables = eventos['LEVEL1'].str
unidad_variables = eventos['valueuom'].str

for nombre, unidad, control, conversion in CONVERSION_UNIDADES:
    indices_variable = nombre_variables.contains(nombre, case=False, na=False)
    necesita_conversion = indices_variable & False
    if unidad is not None:
        necesita_conversion |= nombre_variables.contains(unidad, case=False, na=False) | unidad_variables.contains(unit, case=False, na=False)
    if control is not None:
        necesita_conversion |= control(eventos['value'])
    indice = indices_variable & necesita_conversion
    eventos.loc[indice, 'value'] = conversion(eventos['value'][indice])


# Detección de valores anómalos
 
# Para detectar los valores anómalos asociaremos a cada variable numérica un umbral superior e inferior. 
 
# - Si las variables observadas, caen fuera de estos dos umbrales, entonces las trataremos como valores perdidos. 
 
# - Cualquier valor que en el paso anterior no fuera considerado como un valor anómalo, será comprobado de nuevo en base a los umbrales superior e inferior establecidos para detectar si son fisiológicamente significativos. En el caso de los valores que estén fuera de ese rango, (fisiológicamente lógicos) serán reemplazados por el valor válido más cercano. 


rango_variables = pd.read_csv('variables_range.csv', index_col = None)
rango_variables['LEVEL2'] = rango_variables['LEVEL2'].str.lower()
rango_variables = rango_variables.set_index('LEVEL2')

variables = eventos['LEVEL2']
variables_no_null = ~eventos.value.isnull()
variables = set(variables)
rango_nombres = set(rango_variables.index.values)
rango_nombres = [i.lower() for i in rango_nombres]

for nombre_variable in variables:
    nombre_variable_minus = nombre_variable.lower()
    
    if nombre_variable_minus in rango_nombres:
        outlier_bajo, outlier_alto, valor_valido_bajo, valor_valido_alto = [
            rango_variables.loc[nombre_variable_minus, x] for x in ('OUTLIER BAJO', 'OUTLIER ALTO', 'VALIDO BAJO', 'VALIDO ALTO')
        ]
        
        # Lo primero es encontrar los índices de las valores que necesitamos verificar para ver si son anómalos
        indices_variable = variables_no_null & (variables == nombre_variable)
        
        # Comprobamos la existencia de valores anómalos bajos no extremos. Si existen, los sustituímos por 
        # el valor obtenido de la imputación
        indice_outlier_bajo = (eventos.value < outlier_bajo)
        outliers_no_bajos = ~indice_outlier_bajo & (eventos.value < valor_valido_bajo)
        indice_valor_valido_bajo = indices_variable & outliers_no_bajos
        eventos.loc[indice_valor_valido_bajo, 'value'] = valor_valido_bajo
        
        # Hacemos lo mismo, pero comprobando la existencia de valores anómalos altos no extremos. Si existen, 
        # los sustituímos por el valor obtenido de la imputación
        indice_outlier_alto = (eventos.value > outlier_alto)
        outliers_no_altos = ~indice_outlier_alto & (eventos.value > valor_valido_alto)
        indice_valor_valido_alto = indices_variable & outliers_no_altos
        eventos.loc[indice_valor_valido_alto, 'value'] = valor_valido_alto
        
        # Trataremos los valores que se encuentren fuera del umbral como valores perdidos
        indices_outliers = indices_variable & (indice_outlier_bajo | indice_outlier_alto)
        eventos.loc[indices_outliers, 'value'] = np.nan


# Ordenación de los datos
# Reorganizaremos los datos para agrupar los valores de las variables dentro de cada hora, debido a que la base de datos ofrece marcas de tiempo en unidades de segundos y esto se aplica para cada medida de laboratorio y para el registro de los signos vitales.

# Estableceremos una columna para cada variable
eventos = eventos.set_index(['icustay_id', 'itemid', 'label', 'LEVEL1', 'LEVEL2'])
eventos = eventos.groupby(['icustay_id', 'subject_id', 'hadm_id', 'LEVEL2', 'hours_in'])
eventos = eventos.agg(['mean', 'std', 'count'])
eventos.columns = eventos.columns.droplevel(0)
eventos.columns.names = ['Aggregation Function']
eventos = eventos.unstack(level = 'LEVEL2')
eventos.columns = eventos.columns.reorder_levels(order=['LEVEL2', 'Aggregation Function'])

# Estableceremos una fila para cada hora
rellenar_horas_perdidas = pd.DataFrame([[i, x] for i, y in datos_pacientes['max_hours'].iteritems() for x in range(y+1)],
                                 columns=[datos_pacientes.index.names[0], 'hours_in'])
rellenar_horas_perdidas['tmp'] = np.NaN

df = datos_pacientes.reset_index()[['subject_id', 'hadm_id', 'icustay_id']].join(
     rellenar_horas_perdidas.set_index('icustay_id'), on='icustay_id')
df.set_index(['icustay_id', 'subject_id', 'hadm_id', 'hours_in'], inplace=True)

eventos = eventos.reindex(df.index)
eventos = eventos.sort_index(axis = 0).sort_index(axis = 1)

indice = pd.IndexSlice
eventos.loc[:, indice[:, 'count']] = eventos.loc[:, indice[:, 'count']].fillna(0)

# Almacenamos los cambios realizados
indice = pd.IndexSlice
datos_series_temporales = eventos.loc[:, indice[:, 'mean']]
datos_series_temporales = datos_series_temporales.droplevel('Aggregation Function', axis = 1) 
datos_series_temporales = datos_series_temporales.reset_index() 
datos_series_temporales.to_csv('datos_series_temporales_sin_imputar.csv')


# Imputación de valores perdidos

# Por una parte, un valor podría estar presente para un paciente en particular, pero podría faltar para este período de tiempo en particular. Y la otra opción, es que la variable no se haya medido en absoluto para un paciente concreto. En el primer caso, calcularemos el valor perdido con la media teniendo en cuenta solo ese paciente en particular. Mientras que, en el segundo caso, completaremos el valor perdido con la media teniendo en cuenta todos los pacientes.

indice = pd.IndexSlice
datos_series_temporales = eventos.loc[:, indice[:, ['mean', 'count']]]

# Obtenemos la media de las horas de cada variable para cada paciente
media_estancia_uci = datos_series_temporales.loc[:, indice[:, 'mean']].groupby(['subject_id', 'hadm_id', 'icustay_id']).mean()

# Obtenemos la media global de cada variable
media_global = datos_series_temporales.loc[:, indice[:, 'mean']].mean(axis = 0)

# Rellenamos con "Nan", o con la media de las horas de cada paciente, o con la media global
datos_series_temporales.loc[:, indice[:, 'mean']] = datos_series_temporales.loc[:, indice[:, 'mean']].groupby(
    ['subject_id', 'hadm_id', 'icustay_id']).fillna(method='ffill').groupby(
    ['subject_id', 'hadm_id', 'icustay_id']).fillna(media_estancia_uci).fillna(media_global)

# Creamos una máscara que indique que la variable está presente
datos_series_temporales.loc[:, indice[:, 'count']] = (eventos.loc[:, indice[:, 'count']] > 0).astype(float)
datos_series_temporales.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)

# Añadimos una variable que indique cuándo se hizo la última medición
sin_datos = (1 - datos_series_temporales.loc[:, indice[:, 'mask']])
horas_sin_datos = sin_datos.cumsum()
horas_desde_ultima_medicion = horas_sin_datos - horas_sin_datos[sin_datos==0].fillna(method='ffill')
horas_desde_ultima_medicion.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)
datos_series_temporales = pd.concat((datos_series_temporales, horas_desde_ultima_medicion), axis = 1)
datos_series_temporales.loc[:, indice[:, 'time_since_measured']] = datos_series_temporales.loc[:, indice[:, 'time_since_measured']].fillna(100)
datos_series_temporales.sort_index(axis=1, inplace=True)


# Estandarización de las variables continuas
 
# Existen tanto variables categóricas, como continuas. Las variables continuas necesitan ser estandarizadas para que se encuentren en una misma escala. Podemos hacerlo eliminando la media y dividiendo por la desviación estándar.

def minmax(x):
    minimos = x.min()
    maximos = x.max()
    return ((x - minimos) / (maximos - minimos))


def std_time_since_measurement(x):
    indice = pd.IndexSlice
    x = np.where(x==100, 0, x)
    medias = x.mean()
    stds = x.std() + 0.0001
    return ((x - medias)/stds)

datos_series_temporales.loc[:, indice[variable_continua, 'mean']] = datos_series_temporales.loc[:, indice[variable_continua, 'mean']].apply(lambda x: minmax(x))
datos_series_temporales.loc[:, indice[:, 'time_since_measured']] = datos_series_temporales.loc[:, indice[:, 'time_since_measured']].apply(lambda x: std_time_since_measurement(x))


# Codificación One Hot Encoding para variables categóricas
 
# Los valores categóricos deben ser representados de forma coherente. Esto es necesario para que luego seamos capaces de agregar estos valores con las variables continuas. Para ello, haremos uso de la codificación One Hot Encoding.

# Primero tenemos que redondear las variables categóricas a la categoría más cercana:
datos_categoricos = datos_series_temporales.loc[:, indice[variable_categorica, 'mean']].copy(deep=True)
datos_categoricos = datos_categoricos.round()
codificacion_one_hot = pd.get_dummies(datos_categoricos, variable_categorica)

# Limpiamos las columnas que no necesitamos y añadimos  las codificaciones "dummy":
for col in variable_categorica:
    if col in datos_series_temporales.columns:
        datos_series_temporales.drop(col, axis = 1, inplace=True)
datos_series_temporales.columns = datos_series_temporales.columns.droplevel(-1)
datos_series_temporales = pd.merge(datos_series_temporales.reset_index(), codificacion_one_hot.reset_index(), how='inner', left_on=['subject_id', 'icustay_id', 'hadm_id', 'hours_in'],
                           right_on=['subject_id', 'icustay_id', 'hadm_id', 'hours_in'])
datos_series_temporales = datos_series_temporales.set_index(['subject_id', 'icustay_id', 'hadm_id', 'hours_in'])


# Preprocesamiento de las salidas (Y)

# Primero obtendremos el número de valores "Nan" por cada variable
print(datos.isna().sum())

# Los reemplazaremos con ceros
datos = datos.fillna(0)

# Renombramos las columnas y guardamos los datos preprocesados
s = datos_series_temporales.columns.to_series()
datos_series_temporales.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})

datos_series_temporales.to_hdf('datos_preprocesados.h5', 'X')
datos.to_hdf('datos_preprocesados.h5', 'Y')