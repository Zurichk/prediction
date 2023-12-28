# Paso 1: Importar las bibliotecas necesarias
import pickle
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

# Guardar el modelo en un archivo pickle
with open('modelo.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Cargar el modelo desde el archivo pickle
with open('modelo.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)

# Paso 6: Hacer predicciones con el modelo
y_pred = modelo_cargado.predict(X_test)

# Paso 7: Evaluar el modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Calcular y mostrar la precisión global
accuracy = (confusion_matrix(y_test, y_pred)[
            0, 0] + confusion_matrix(y_test, y_pred)[1, 1]) / len(y_test)
print(f"Precisión Global: {accuracy}")
