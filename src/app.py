from utils import db_connect
engine = db_connect()

# your code here
# Explore here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
df_banco = pd.read_csv('../data/raw/bank-marketing-campaign-data.csv', sep=';')
print(df_banco)
# ESTADÍSTICAS DESCRIPTIVAS DE LA BASE DE DATOS:

df_banco.describe()
# INFORMACIÓN DE LA BASE DE DATOS:

df_banco.info()
# NUBE DE PUNTOS Y RECTA DE REGRESIÓN:

df_banco['age'] = pd.to_numeric(df_banco['age'], errors='coerce')
df_banco['y'] = df_banco['y'].map({'no': 0, 'yes': 1})

plt.figure(figsize=(8, 5))
sns.regplot(x='age', y='y', data=df_banco)
plt.title("Relación entre la edad y la contratación del depósito a largo plazo")
plt.xlabel("Edad")
plt.ylabel("Probabilidad de contratación (y)")
plt.show()
print("Como se puede observar en el gráfico, la edad no es un condicionante para la contratación del fondo")
# REVISIÓN DE VALORES NULOS:

print(df_banco.isnull().sum())

# DISTRIBUCIÓN DE LA VARIABLE:

plt.figure(figsize=(8, 5))
sns.countplot(x='y', data=df_banco, color='indigo', edgecolor='black')
plt.show()
# DISTRIBUCIÓN DE LAS VARIABLES CATEGÓRICAS:

variables_categoricas = ['job', 'education', 'housing', 'loan']
for var in variables_categoricas:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_banco, x=var, hue='y', palette=['indigo', 'darkblue'], edgecolor='black')
    plt.title(f"{var.capitalize()} vs. Deposit Contracting")
    plt.xticks(rotation=45)
    plt.show()
# CORRELACIÓN NUMÉRICA:

df_banco_corr = df_banco.select_dtypes(include=['float64', 'int64']).corr()

plt.figure(figsize=(8, 5))
sns.heatmap(df_banco_corr[['y']].sort_values(by='y', ascending=False), annot=True, cmap='coolwarm')
plt.title("Correlación de Variables")
plt.show()
print("La variable de duración de la llamada es la que más correlación tiene con la variable y. Esto es normal teniendo en cuanto que cuanto más tiempo estén con el cliente en llamada es más probable que contrate el servicio. La segunda variable que tiene más correlación con y es previous, lo que también es normal ya que si anteriormente ya contrataron es probable que vuelvan a contratar nuevamente. Y la variable con menor correlación con y es nr.employed; esto tiene sentido ya que no normalmente no importa quien sea el empleado quien les ofrezca el servicio.")
PREPROCESAMIENTO DE DATOS (EDA):
df_banco_encoded = pd.get_dummies(df_banco, drop_first=True)
print(df_banco_encoded)
# SEPARACIÓN X e y:

X = df_banco_encoded.drop('y', axis=1)
y = df_banco_encoded['y']
# TRAIN TEST:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# ESTANDARIZACIÓN DE VARIABLES:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
REGRESIÓN LOGÍSTICA:
# ENTRENAMIENTO DEL MODELO DE REGRESIÓN:

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# PEDICCIONES:

y_pred = log_reg.predict(X_test_scaled)
# EVALUACIÓN DEL MODELO:

print(f"Matriz de Confusión: {confusion_matrix(y_test, y_pred)}")

print(f"\nReporte de Clasificación: {classification_report(y_test, y_pred)}")

print(f"\nPrecisión: {accuracy_score(y_test, y_pred):4f}")
print("Aunque el modelo tiene un buen desempeño (91% de precisión) en su mayoría lo que muestra es a los clientes que no contratan el servicio ofrecido por el banco. También parece que detecta mal a los que si contratan el servicio, cosa que podría ser un fallo del banco a la hora de ofrecer sus servicios a los clientes potenciales equivocados. ")
REGULARIZACIÓN LOGÍSTICA:
# MODELO DE REGULARIZACIÓN:

log_reg_cv = LogisticRegressionCV(
    cv=5,
    penalty='l2',
    scoring='f1',
    class_weight='balanced',
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)

# ENTRENAMIENTO:

log_reg_cv.fit(X_train_scaled, y_train)

# PREDICCIONES:

y_pred_cv = log_reg_cv.predict(X_test_scaled)
# EVALUACIÓN DEL MODELO OPTIMIZADO:

print(f"Matriz de Confusión: {confusion_matrix(y_test, y_pred_cv)}")

print(f"\nReporte de Clasificación: {classification_report(y_test, y_pred_cv)}")

print(f"\nPrecisión: {accuracy_score(y_test, y_pred_cv):4f}")
print("La precisión ha bajado de un 91 a un 86%, mejorándo drásticamente la capacidad de detectar clientes que si contratarán el servicio bancario. Tras construir y optimizar el modelo de regresión logística para detectar qué clientes contratarían un depósito bancario a largo plazo; Con todo esto conseguimos la detección del tipo de clientes que si estarían interesados en la contratación del servicio. El recall ha subido del 47 al 90% con la optimización; convirtiéndolo en una herramienta eficaz para enfocar los esfuerzos del Departamento de Marketing hacia perfiles de clientes con mayor afinidad a la contratación de dicho servicio.")