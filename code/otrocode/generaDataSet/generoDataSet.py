import pandas as pd
import numpy as np

rangoinf = 1
rangosup = 5001

# Definir las opciones para cada columna
ubicaciones = ['Lat, Long'] * (rangosup - rangoinf)
provincias = ['Álava', 'Albacete', 'Alicante', 'Almería', 'Asturias', 'Ávila', 'Badajoz', 'Baleares', 'Barcelona', 'Biscay', 'Burgos', 'Cáceres', 'Cádiz', 'Cantabria', 'Castellón', 'Ciudad Real', 'Córdoba', 'Cuenca', 'Girona', 'Granada', 'Guadalajara', 'Guipúzcoa', 'Huelva', 'Huesca', 'Jaén',
              'La Coruña', 'La Rioja', 'Las Palmas', 'León', 'Lleida', 'Lugo', 'Madrid', 'Málaga', 'Murcia', 'Navarra', 'Ourense', 'Palencia', 'Pontevedra', 'Salamanca', 'Santa Cruz de Tenerife', 'Segovia', 'Sevilla', 'Soria', 'Tarragona', 'Teruel', 'Toledo', 'Valencia', 'Valladolid', 'Zamora', 'Zaragoza']
# 100 para 'Alto', 75 para 'Medio', 50 para 'Bajo'
interes_turistico = [100, 75, 50, 25, 0]
acceso_transporte = [1, 0]  # 1 para 'Si', 0 para 'No'
zona_residencial = [1, 0]  # 1 para 'Si', 0 para 'No'
zona_comercial = [1, 0]  # 1 para 'Si', 0 para 'No'
# 100 para 'MuyAlto', 75 para 'Alto', 50 para 'Medio', 25 para 'Bajo', 0 para 'Nulo'
indice_criminalidad = [100, 75, 50, 25, 0]
# calificacion_ambiental = ['A', 'B', 'C']
# np.inf para 'Ninguna', 20 para 'Altura Maxima 20m', 15 para 'Altura Maxima 15m'
restricciones_construccion = [np.inf, 20, 15]

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame({
    'ID': range(rangoinf, rangosup),
    'Ubicacion': ubicaciones,
    'Localidad': np.random.choice(provincias, rangosup - rangoinf),
    'Interes Turistico': np.random.choice(interes_turistico, rangosup - rangoinf),
    'Superficie (m²)': np.random.randint(500, 1000, rangosup - rangoinf),
    'Precio (€)': np.random.randint(200000, 400000, rangosup - rangoinf),
    'Acceso a Transporte Publico': np.random.choice(acceso_transporte, rangosup - rangoinf),
    'Proximidad a Servicios (km)': np.random.uniform(0.1, 1.0, rangosup - rangoinf),
    'Zona Residencial': np.random.choice(zona_residencial, rangosup - rangoinf),
    'Zona Comercial': np.random.choice(zona_comercial, rangosup - rangoinf),
    'Indice de Criminalidad': np.random.choice(indice_criminalidad, rangosup - rangoinf),
    'Restricciones de Construccion': np.random.choice(restricciones_construccion, rangosup - rangoinf),
})

# Definir una función para determinar si comprar o no la parcela


def comprar_o_no(fila):
    return (
        (fila['Precio (€)'] < 300000) and
        (fila['Interes Turistico'] >= 50) and
        (fila['Proximidad a Servicios (km)'] <= 0.9)
    )


# Aplicar la función a cada fila del DataFrame para crear la nueva columna 'Comprar'
df['Comprar'] = df.apply(comprar_o_no, axis=1)

# Guardar el DataFrame actualizado como un archivo CSV
df.to_csv('estudio_viabilidad.csv', sep=';', index=False)
