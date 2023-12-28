# Paso 1: Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

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
# Significa que el 20% del conjunto de datos se reservará para pruebas, y el 80% se utilizará para entrenamiento.

# Paso 4: Estandarizar los datos
# Solo necesitas estandarizar las columnas numéricas
num_cols = ['Superficie (m²)', 'Precio (€)', 'Proximidad a Servicios (km)',
            'Indice de Criminalidad', 'Interes Turistico', 'Acceso a Transporte Publico', 'Zona Residencial', 'Zona Comercial']
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Paso 5: Entrenar el modelo de clasificación
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# NOTA:::::::::::: Al subir n_neighbors a 10:
# Suavización de la frontera de decisión: El modelo se volverá más suave y menos propenso a sobreajustarse a pequeñas variaciones en los datos de entrenamiento.
# Robustez ante ruido: El modelo será menos sensible a datos atípicos o ruido en el conjunto de entrenamiento, ya que la influencia de un solo vecino es menor.
# Posible pérdida de detalles finos: A medida que aumentas n_neighbors, el modelo puede perder la capacidad de capturar patrones finos y detalles en los datos, ya que la decisión se basa en un número mayor de vecinos.

# Paso 6: Hacer predicciones con el modelo
y_pred = classifier.predict(X_test)

# Paso 7: Evaluar el modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Calcular y mostrar la precisión global
accuracy = (confusion_matrix(y_test, y_pred)[
            0, 0] + confusion_matrix(y_test, y_pred)[1, 1]) / len(y_test)
print(f"Precisión Global: {accuracy}")

########################################################################################

# Probamos con un  `df_nuevo` con las características de las nuevas parcelas
data_nuevo = {
    'Interes Turistico': [85],
    'Superficie (m²)': [750],
    'Precio (€)': [250000],
    'Acceso a Transporte Publico': [1],
    'Proximidad a Servicios (km)': [0.9],
    'Zona Residencial': [1],
    'Zona Comercial': [0],
    'Indice de Criminalidad': [25],
}

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

for i, datos in enumerate(conjuntos_de_datos, 1):
    # # Crear un DataFrame de pandas con los datos
    df_nuevo = pd.DataFrame(conjuntos_de_datos[i-1])
    # Asegúrate de que las columnas en `df_nuevo` están en el mismo orden que en `X_train`
    df_nuevo = df_nuevo[X_train.columns]

    # Estandarizar las características de las nuevas parcelas
    df_nuevo[num_cols] = scaler.transform(df_nuevo[num_cols])

    # Usar el modelo para hacer predicciones
    predicciones = classifier.predict(df_nuevo)
    print("\nResultados de Predicciones para Nuevas Parcelas:")
    print(predicciones)

# # Crear un DataFrame de pandas con los datos
# df_nuevo = pd.DataFrame(conjuntos_de_datos[4])

# # Asegúrate de que las columnas en `df_nuevo` están en el mismo orden que en `X_train`
# df_nuevo = df_nuevo[X_train.columns]

# # Estandarizar las características de las nuevas parcelas
# df_nuevo[num_cols] = scaler.transform(df_nuevo[num_cols])

# # Usar el modelo para hacer predicciones
# predicciones = classifier.predict(df_nuevo)

# # Mostrar el resultado
# print("\nResultados de Predicciones para Nuevas Parcelas:")
# print(predicciones)

# Este código calculará la proporción de vecinos más cercanos que pertenecen a la clase 1 para cada conjunto de datos de nuevas parcelas.
# Aunque no es un porcentaje directo de certeza, puede darte una idea de la "confianza" del modelo en sus predicciones.
# Un valor cercano a 1 indicaría una mayor confianza en la predicción de la clase 1.
for i, datos in enumerate(conjuntos_de_datos, 1):
    # Crear un DataFrame de pandas con los datos
    df_nuevo = pd.DataFrame(conjuntos_de_datos[i-1])
    # Asegúrate de que las columnas en `df_nuevo` están en el mismo orden que en `X_train`
    df_nuevo = df_nuevo[X_train.columns]

    # Estandarizar las características de las nuevas parcelas
    df_nuevo[num_cols] = scaler.transform(df_nuevo[num_cols])

    # Obtener las distancias y los índices de los k vecinos más cercanos
    distancias, indices = classifier.kneighbors(df_nuevo)

    # Contar la cantidad de vecinos que votaron por la clase 1
    votos_clase_1 = sum(y_train.iloc[indices[0]])

    # Calcular la proporción de votos por la clase 1
    proporcion_votos_clase_1 = votos_clase_1 / len(indices[0])

    print("\nProporción de Votos para Nuevas Parcelas:")
    print(proporcion_votos_clase_1)
