# Paso 1: Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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

# Paso 5: Entrenar el modelo XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Paso 6: Hacer predicciones con el modelo
y_pred = xgb_model.predict(X_test)

# Paso 7: Evaluar el modelo
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# Calcular y mostrar la precisión global
accuracy = (confusion_matrix(y_test, y_pred)[
            0, 0] + confusion_matrix(y_test, y_pred)[1, 1]) / len(y_test)
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

umbral = 0.5  # Puedes ajustar este umbral según tus necesidades

print("\n--------------------------------------------------------------")
for i, datos in enumerate(conjuntos_de_datos, 1):
    # Crear un DataFrame de pandas con los datos
    df_nuevo = pd.DataFrame(conjuntos_de_datos[i-1])
    # Asegúrate de que las columnas en `df_nuevo` están en el mismo orden que en `X_train`

    # Estandarizar las características de las nuevas parcelas
    df_nuevo[num_cols] = scaler.transform(df_nuevo[num_cols])

    # Usar el modelo para hacer predicciones
    predicciones = xgb_model.predict(df_nuevo)
    predicciones_probabilidades = xgb_model.predict_proba(
        df_nuevo)[:, 1]  # Probabilidad de la clase positiva
    predicciones_binarias = (predicciones_probabilidades >= umbral).astype(int)

    print(
        f"\nResultados de Predicciones para Nuevas Parcelas (XGBoost - Conjunto {i}):")
    print(predicciones)
    print(np.round(predicciones_probabilidades, 2))
    print(f"Predicción Binaria (Umbral={umbral}): {predicciones_binarias}")
