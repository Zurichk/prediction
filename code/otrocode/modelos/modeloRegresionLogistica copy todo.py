# Paso 1: Importar las bibliotecas necesarias
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

# Paso 6: Hacer predicciones con el modelo
y_pred = regressor.predict(X_test)
# Obtener probabilidades de predicción
y_pred_prob = regressor.predict_proba(X_test)[:, 1]

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

umbral = 0.64  # Puedes ajustar este umbral según tus necesidades

print("\n--------------------------------------------------------------")
for i, datos in enumerate(conjuntos_de_datos, 1):
    # Crear un DataFrame de pandas con los datos
    df_nuevo = pd.DataFrame(conjuntos_de_datos[i-1])
    # Asegúrate de que las columnas en `df_nuevo` están en el mismo orden que en `X_train`
    df_nuevo = df_nuevo[X_train.columns]

    # Estandarizar las características de las nuevas parcelas
    df_nuevo[num_cols] = scaler.transform(df_nuevo[num_cols])

    # Usar el modelo para hacer predicciones
    predicciones = regressor.predict(df_nuevo)
    predicciones_probabilidades = regressor.predict_proba(
        df_nuevo)[:, 1]  # Probabilidad de la clase positiva
    predicciones_binarias = (predicciones_probabilidades >= umbral).astype(int)

    print(
        f"\nResultados de Predicciones para Nuevas Parcelas (Regresión Logística - Conjunto {i}):")
    print(predicciones)
    print(np.round(predicciones_probabilidades, 2))
    print(f"Predicción Binaria (Umbral={umbral}): {predicciones_binarias}")


# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title('Matriz de Confusión')
plt.show()

# Visualizar comparación de características
if hasattr(regressor, 'coef_'):
    feature_importance = regressor.coef_[0]
    features = X_train.columns
    sns.barplot(x=feature_importance, y=features)
    plt.title('Importancia de las Características')
    plt.show()

# Análisis de umbral dinámico (ejemplo)
thresholds = np.arange(0.2, 1.0, 0.1)
for threshold in thresholds:
    y_pred_threshold = (y_pred_prob >= threshold).astype(int)
    print(f"Umbral: {threshold}")
    print(classification_report(y_test, y_pred_threshold))
