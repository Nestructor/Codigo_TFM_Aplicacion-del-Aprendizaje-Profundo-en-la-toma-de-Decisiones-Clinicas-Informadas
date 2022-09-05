# Extracción de datos en MIMIC-III


# - Código 'Extraccion_Datos.py'   
# - Trabajo Fin de Máster.   
# - Néstor Bolaños Bolaños. (nestorbolanos@correo.ugr.es)

# Bibliotecas


# Instalación del módulo pyyaml en caso necesario
# !pip3 install pyyaml
import warnings
warnings.filterwarnings('ignore')
import psycopg2
import numpy as np
import pandas as pd
import os
import yaml


# Conexión con MIMIC-III

usuario = 'postgres'
contrasena='postgres'
base_de_datos = 'mimic'
esquema = 'public, mimic, mimiciii;'

conexion = psycopg2.connect(base_de_datos, usuario, contrasena)
cur = conexion.cursor()
cur.execute('SET search_path to ' + esquema)


# Predicción de la mortalidad intrahospitalaria

# Vamos a centrarnos en la predicción de la mortalidad intrahospitalaria. Seleccionaremos todos los pacientes en MIMIC-III que cumplan las siguientes condiciones:
 
# - Pacientes que han estado al menos más de un día en la UCI y menos de 10 días de estancia.
# - Solo pacientes adultos con edad mayor o igual a 15 años
 
# Para obtener los datos necesarios, realizaremos una consulta que cumpla con estas condiciones. El resultado esta consulta consistirá en un conjunto de datos manejable formado por 30063 filas y 24 columnas.


edad = 15
npacientes = 0 # si queremos realizar la consulta con un número determinado de pacientes
if npacientes > 0:
    limite = 'LÍMITE: ' + str(npacientes)
else:
    limite = ''


"""
Identificador paciente: subject_id
Ingreso paciente: admittime
Ingreso hospitalario: hospstay_seq
Identificador Hora de Ingreso: hadm_id
Hora de Ingreso: intime
Hora de Alta: outtime
"""

consulta = """
with patient_and_icustay_details as (
    SELECT distinct
        p.gender, p.dob, p.dod, s.*, a.admittime, a.dischtime, a.deathtime, a.ethnicity, a.diagnosis,
        DATE_PART('year', s.intime) - DATE_PART('year', p.dob) as admission_age,
        DATE_PART('day', s.outtime - s.intime) as los_icu
    FROM patients p 
    WHERE s.first_careunit NOT like 'NICU'
        and s.hadm_id is not null and s.icustay_id is not null
    ORDER BY s.subject_id 
)
SELECT * 
FROM patient_and_icustay_details 
WHERE hospstay_seq = 1
    and icustay_seq = 1
    and admission_age >=  """ + str(edad) + """
    and los_icu >= 0.5
""" + str(limite)

datos_pacientes = pd.read_sql_query('SET search_path to ' + esquema + consulta, conexion)

datos_pacientes


# Extracción de variables

# De las 24 variables obtenidas en la consulta, nos quedaremos con las 17 clínicamente más significativas. Realizaremos una consulta obteniendo previamente la información necesaria haciendo uso de las tablas de diccionarios, chartevents e icustays.


variables = ('Capillary refill rate', 'Diastolic blood pressure', 'Fraction inspired oxygen',
             'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total',
             'Glascow coma scale verbal response', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure',
             'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH')

mapa_variables = pd.read_csv('itemid_to_variable_map.csv')

ids_uci = datos_pacientes['icustay_id']
ids_uci = tuple(set([str(i) for i in ids_uci]))
pacientes = datos_pacientes['subject_id']
pacientes = tuple(set([str(i) for i in pacientes]))
admisiones_hospital = datos_pacientes['hadm_id']
admisiones_hospital = tuple(set([str(i) for i in admisiones_hospital]))

lab_items = []
chart_items = []
for i in range(mapa_variables.shape[0]):
    if mapa_variables['LEVEL2'][i] in variables:
        if mapa_variables['LINKSTO'][i] == 'chartevents':
            chart_items.append(mapa_variables['ITEMID'][i])
        elif mapa_variables['LINKSTO'][i] == 'labevents':
            lab_items.append(mapa_variables['ITEMID'][i])
            
chart_lab_items = chart_items + lab_items
mapa_variables = mapa_variables[mapa_variables.ITEMID.isin(chart_lab_items)]
chart_items = tuple(set([str(i) for i in chart_items]))
lab_items = tuple(set([str(i) for i in lab_items]))

consulta = """
SELECT c.subject_id, i.hadm_id, c.icustay_id, c.charttime, c.itemid, c.value, c.valueuom
FROM icustays i
where c.icustay_id in """ + str(ids_uci) + """
  and c.itemid in """ + str(chart_items) + """
  and c.valuenum is not null
SELECT distinct i.subject_id, i.hadm_id, i.icustay_id, l.charttime, l.itemid, l.value, l.valueuom
FROM icustays i
where i.icustay_id in """ + str(ids_uci) + """
  and l.itemid in """ + str(lab_items) + """
  and l.valuenum > 0 -- las pruebas de laboratorio no pueden ser <= 0
"""

eventos = pd.read_sql_query('SET search_path to ' + esquema + consulta, conexion)

id_items = tuple(set(eventos.itemid.astype(str)))
eventos.head()


consulta_d_items = \
        """
        SELECT itemid, label, dbsource, linksto, category, unitname
        FROM d_items
        WHERE itemid in """ + str(id_items)

salida_d = pd.read_sql_query('SET search_path to ' + esquema + consulta_d_items, conexion)


# Eliminamos el texto de las variables categóricas (escala de coma de Glasgow) para convertirlas en numéricas:

diccionario = {'4 Spontaneously': '4', '3 To speech': '3', '2 To pain': '2', '1 No Response': '1',
                         '5 Oriented': '5', '1.0 ET/Trach': '1', '4 Confused': '4', '2 Incomp sounds': '2', 
                         '3 Inapprop words': '3', 'Spontaneously': '4', 'To Speech': '3', 'None': '1', 'To Pain': '2',
                         '6 Obeys Commands': '6', '5 Localizes Pain': '5', '4 Flex-withdraws': '4', '2 Abnorm extensn': '2',
                         '3 Abnorm flexion': '3', 'No Response-ETT': '1', 'Oriented': '5', 'Confused': '4', 
                         'No Response': '1', 'Incomprehensible sounds': '2', 'Inappropriate Words': '3', 
                         'Obeys Commands': '6', 'No response': '1', 'Localizes Pain': '5', 'Flex-withdraws': '4',
                         'Abnormal extension': '2', 'Abnormal flexion': '3', 'Abnormal Flexion': '3', 
                          'Abnormal Extension': '2'}

for k, value in diccionario.items():
    eventos['value'] = eventos['value'].replace(k, value) 


# Cambiamos los tipos de datos estableciendo los índices
eventos['value'] = pd.to_numeric(eventos['value']) #, 'coerce')
eventos = eventos.astype({k: int for k in ['subject_id', 'hadm_id', 'icustay_id']})
datos_pacientes = datos_pacientes.reset_index().set_index('icustay_id')
mapa_variables = mapa_variables[['LEVEL2', 'ITEMID', 'LEVEL1']].rename(
    {'LEVEL2': 'LEVEL2', 'LEVEL1': 'LEVEL1', 'ITEMID': 'itemid'}, axis=1).set_index('itemid')

# Necesitamos los datos en horas
horas = lambda x: max(0, x.days*24 + x.seconds // 3600)
eventos = eventos.set_index('icustay_id').join(datos_pacientes[['intime']])
eventos['hours_in'] = (eventos['charttime'] - eventos['intime']).apply(horas)
eventos = eventos.drop(columns=['charttime', 'intime']) 

# Unimos el resultado de la consulta agrupando las variables
eventos = eventos.set_index('itemid', append=True)
eventos = eventos.join(mapa_variables)
salida_d = salida_d.set_index('itemid')
eventos = eventos.join(salida_d) 
eventos = eventos.set_index(['label', 'LEVEL1', 'LEVEL2'], append=True)
datos_pacientes['max_hours'] = (datos_pacientes['outtime'] - datos_pacientes['intime']).apply(horas)


# Almacenamos los resultados obtenidos:
np.save('pacientes.npy', datos_pacientes['subject_id'])
np.save('tiempo_horas.npy', datos_pacientes['max_hours'])
datos_pacientes.to_hdf('datos_extraidos.h5', 'datos_pacientes')
eventos.to_hdf('datos_por_hora.h5', 'X')


# Extracción del tiempo de estancia y de la mortalidad intrahospitalaria
 
# Nota: entendemos la mortalidad hospitalaria como el fallecimiento de un paciente después de ser ingresado en el hospital y antes de su hora de alta.


datos_extraidos = pd.DataFrame(datos_pacientes.index)

# Mortalidad
mortalidad = datos_pacientes.dod.notnull() & ((datos_pacientes.admittime <= datos_pacientes.dod) & (datos_pacientes.outtime >= datos_pacientes.dod))
mortalidad = mortalidad | (datos_pacientes.deathtime.notnull() & ((datos_pacientes.admittime <= datos_pacientes.deathtime) &  (datos_pacientes.dischtime >= datos_pacientes.deathtime)))

datos_extraidos['in_hospital_mortality'] = mortalidad.astype(int)

# Estancia
datos_extraidos['los'] = datos_pacientes['los'] * 24.0
datos_extraidos.to_hdf('datos_extraidos.h5', 'Y')