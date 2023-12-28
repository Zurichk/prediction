import pickle
import modeloClasificacion

# Importar el modelo desde modeloClasificacion
classifier = modeloClasificacion.classifier

# Guardar el modelo en un archivo pickle
with open('modelo.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Cargar el modelo desde el archivo pickle
with open('modelo.pkl', 'rb') as f:
    modelo_cargado = pickle.load(f)
