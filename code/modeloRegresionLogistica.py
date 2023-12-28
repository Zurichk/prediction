# Paso 1: Importar las bibliotecas necesarias
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# Paso 2: Cargar los datos
# Asegúrate de que la ruta al archivo CSV sea la correcta
df = pd.read_csv('estudio_viabilidad.csv', sep=';')

# Supongamos que tienes una columna 'Comprar' que es 1 si decidiste comprar la parcela y 0 si no
y = df['Comprar']
X = df.drop(['Comprar', 'Restricciones de Construccion',
             'Localidad', 'Ubicacion', 'ID'], axis=1)

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Paso 4: Estandarizar los datos
# Solo necesitas estandarizar las columnas numéricas
num_cols = ['Superficie (m²)', 'Precio (€)', 'Proximidad a Servicios (km)',
            'Indice de Criminalidad', 'Interes Turistico', 'Acceso a Transporte Publico', 'Zona Residencial', 'Zona Comercial']
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Paso 5: Entrenar el modelo de regresión logística
regressor = LogisticRegression()
regressor.fit(X_train, y_train)

# Guardar el modelo en un archivo pickle
with open('code/modelo/modelo.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Guardar el estandarizador en un archivo pickle
with open('code/modelo/estandarizador.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Cargar el modelo desde el archivo pickle
with open('code/modelo/modelo.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

# Paso 6: Hacer predicciones con el modelo
y_pred = modelo_cargado.predict(X_test)
# Obtener probabilidades de predicción
y_pred_prob = modelo_cargado.predict_proba(X_test)[:, 1]

# Paso 7: Evaluar el modelo
print("Matriz de Confusión:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nInforme de Clasificación:")
class_report = classification_report(y_test, y_pred)
print(class_report)

# Calcular y mostrar la precisión global
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / len(y_test)
print(f"Precisión Global: {accuracy}")

umbral = 0.64  # Puedes ajustar este umbral según tus necesidades

# Paso 8: Predicciones con Nuevos Datos
conjuntos_de_datos = [
    {
        'Interes Turistico': [95],
        'Superficie (m²)': [800],
        'Precio (€)': [300000],
        'Acceso a Transporte Publico': [1],
        'Proximidad a Servicios (km)': [0.8],
        'Zona Residencial': [1],
        'Zona Comercial': [0],
        'Indice de Criminalidad': [20],
    },
    {
        'Interes Turistico': [80],
        'Superficie (m²)': [700],
        'Precio (€)': [220000],
        'Acceso a Transporte Publico': [1],
        'Proximidad a Servicios (km)': [0.5],
        'Zona Residencial': [1],
        'Zona Comercial': [1],
        'Indice de Criminalidad': [30],
    },
    {
        'Interes Turistico': [90],
        'Superficie (m²)': [850],
        'Precio (€)': [280000],
        'Acceso a Transporte Publico': [1],
        'Proximidad a Servicios (km)': [1.0],
        'Zona Residencial': [1],
        'Zona Comercial': [0],
        'Indice de Criminalidad': [15],
    },
    {
        'Interes Turistico': [85],
        'Superficie (m²)': [600],
        'Precio (€)': [350000],
        'Acceso a Transporte Publico': [1],
        'Proximidad a Servicios (km)': [0.7],
        'Zona Residencial': [1],
        'Zona Comercial': [0],
        'Indice de Criminalidad': [25],
    },
    {
        'Interes Turistico': [80],
        'Superficie (m²)': [720],
        'Precio (€)': [260000],
        'Acceso a Transporte Publico': [1],
        'Proximidad a Servicios (km)': [0.9],
        'Zona Residencial': [0],
        'Zona Comercial': [1],
        'Indice de Criminalidad': [22],
    }
]

# Convertir el conjunto de datos a DataFrame
X_nuevos = pd.DataFrame(conjuntos_de_datos)


print("\n--------------------------------------------------------------")
# Continuación del Paso 8: Predicciones con Nuevos Datos
for i, datos in enumerate(conjuntos_de_datos):
    X_nuevos = pd.DataFrame(datos)

    # Estandarizar los datos
    X_nuevos[num_cols] = scaler.transform(X_nuevos[num_cols])

    # Hacer la predicción
    y_pred_nuevos = modelo_cargado.predict(X_nuevos)
    y_pred_prob_nuevos = modelo_cargado.predict_proba(X_nuevos)[:, 1]

    # Imprimir la predicción
    print(f"\nConjunto de datos {i + 1}:")
    print(f"Predicción: {'Comprar' if y_pred_nuevos[0] else 'No comprar'}")
    print(f"Probabilidad de comprar: {y_pred_prob_nuevos[0]}")

print("\n--------------------------------------------------------------")
print(X_nuevos.to_string(index=False))


# # Obtener los nombres de las características
# feature_names = X_train.columns

# # Obtener los coeficientes del modelo
# coefficients = modelo_cargado.coef_[0]

# # Crear un DataFrame para visualizar los coeficientes
# coef_df = pd.DataFrame(
#     {'Caracteristica': feature_names, 'Coeficiente': coefficients})

# # Ordenar el DataFrame por el valor absoluto de los coeficientes
# coef_df = coef_df.sort_values(by='Coeficiente', key=abs, ascending=False)

# print(coef_df)
