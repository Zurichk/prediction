import json
from flask import Flask, request, jsonify, render_template, abort, send_file
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
import os

num_cols = ['Superficie (m²)', 'Precio (€)', 'Proximidad a Servicios (km)',
            'Indice de Criminalidad', 'Interes Turistico', 'Acceso a Transporte Publico', 'Zona Residencial', 'Zona Comercial']

# Definir la ruta donde se guardarán los archivos cargados
if os.environ.get('DOCKER', '') == "yes":
    UPLOAD_FOLDER = '/usr/src/app/subidas'
    MODELO_FOLDER = '/usr/src/app/modelo'
    with open(MODELO_FOLDER + '/modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open(MODELO_FOLDER + '/estandarizador.pkl', 'rb') as f:
        estandarizador = pickle.load(f)
else:
    UPLOAD_FOLDER = 'subidas'
    # Cargar el modelo
    with open('modelo/modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)

    with open('modelo/estandarizador.pkl', 'rb') as f:
        estandarizador = pickle.load(f)

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template('home.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    # Comprobar si la petición tiene el archivo
    if 'filejson' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400
    file = request.files['filejson']

    # Si el usuario no selecciona un archivo, el navegador podría
    # enviar una petición sin archivo, así que hay que comprobarlo
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    # Guardar el archivo
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Leer los datos del archivo
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r', encoding='utf-8') as f:
            datos = json.load(f)
    except FileNotFoundError:
        return jsonify({'error': 'Archivo no encontrado.'}), 400

    # Corregir los nombres de las características
    for dic in datos:
        if 'Precio (â‚¬)' in dic:
            dic['Precio (€)'] = dic.pop('Precio (â‚¬)')
        if 'Superficie (mÂ²)' in dic:
            dic['Superficie (m²)'] = dic.pop('Superficie (mÂ²)')

    # Validar los datos (aquí podrías agregar más validaciones)
    if not isinstance(datos, list):
        return jsonify({'error': 'Los datos deben ser una lista.'}), 400

    # Convertir los datos a DataFrame
    X = pd.DataFrame(datos)
    X[num_cols] = estandarizador.transform(X[num_cols])
    print(X.to_string(index=False))

    # Hacer la predicción
    try:
        y_pred = modelo.predict(X)
        y_pred_prob_nuevos = modelo.predict_proba(X)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    prediccion = 'Comprar' if y_pred[0] else 'No comprar'
    # Probabilidad de la clase predicha
    probabilidad = y_pred_prob_nuevos[0][int(y_pred[0])] * 100
    print({'prediccion': prediccion, 'probabilidad': probabilidad})
    print(y_pred)
    # Redirigir a la página result.html con la predicción
    return render_template('results.html', prediction=y_pred.tolist())


@app.route('/enviar_json_bc', methods=['POST'])
def enviar_json_bc():
    # Comprobar si la petición tiene los datos en formato JSON
    if not request.is_json:
        return jsonify({'error': 'No JSON object in the request.'}), 400

    # Leer los datos del JSON
    datos = request.get_json()

    # Corregir los nombres de las características
    for dic in datos:
        if 'Precio (â‚¬)' in dic:
            dic['Precio (€)'] = dic.pop('Precio (â‚¬)')
        if 'Superficie (mÂ²)' in dic:
            dic['Superficie (m²)'] = dic.pop('Superficie (mÂ²)')

    # Validar los datos (aquí podrías agregar más validaciones)
    if not isinstance(datos, list):
        return jsonify({'error': 'Los datos deben ser una lista.'}), 400

    # Convertir los datos a DataFrame
    X = pd.DataFrame(datos)
    X[num_cols] = estandarizador.transform(X[num_cols])
    print(X.to_string(index=False))

    # Hacer la predicción
    try:
        y_pred = modelo.predict(X)
        y_pred_prob_nuevos = modelo.predict_proba(X)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    prediccion = 'Comprar' if y_pred[0] else 'No comprar'
    # Probabilidad de la clase predicha
    probabilidad = y_pred_prob_nuevos[0][int(y_pred[0])] * 100
    print({'prediccion': prediccion, 'probabilidad': probabilidad})
    print(y_pred)
    # Redirigir a la página result.html con la predicción
    return render_template('results.html', prediction=y_pred.tolist())

# Crear una ruta para descargar el fichero generado


@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        print("Error en la descarga:", e)
        abort(500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Ejecutar la aplicación
