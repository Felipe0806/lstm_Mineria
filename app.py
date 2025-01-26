from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import psycopg2
from psycopg2 import pool

app = Flask(__name__)

# Cargar el modelo entrenado
nombre_modelo = 'lstm_compras.h5'
modelo = load_model(nombre_modelo)

# Configurar pool de conexiones PostgreSQL
DB_CONFIG = {
    'dbname': 'mineriadata',
    'user': 'grupo2',
    'password': 'root',
    'host': '54.152.113.164',
    'port': '5432'
}

# Crear un pool de conexiones
conexion_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **DB_CONFIG)

# Preprocesar datos para el modelo
def preprocesar_datos(datos, pasos_históricos=3):
    scaler = MinMaxScaler(feature_range=(0, 1))
    datos_norm = scaler.fit_transform(np.array(datos).reshape(-1, 1))
    
    X = []
    for i in range(len(datos_norm) - pasos_históricos):
        X.append(datos_norm[i:i + pasos_históricos])
    X = np.array(X)
    X = X.reshape((X.shape[0], pasos_históricos, 1))
    
    return X, scaler

# Endpoint para predecir compras futuras de una empresa
@app.route('/predecir_empresa', methods=['POST'])
def predecir_empresa():
    try:
        # Obtener parámetros de la solicitud
        request_data = request.get_json()
        empresa_id = request_data.get('empresa_id')
        pasos_futuros = request_data.get('pasos_futuros', 1)  # Default: 1 predicción futura
        
        if not empresa_id:
            return jsonify({'error': 'Se requiere el ID de la empresa'}), 400
        
        # Obtener conexión del pool
        conn = conexion_pool.getconn()
        try:
            with conn.cursor() as cursor:
                # Consulta para obtener datos de compras
                query = "SELECT fecha, monto FROM compras WHERE empresa = %s ORDER BY fecha ASC"
                cursor.execute(query, (empresa_id,))
                datos_empresa = cursor.fetchall()
            
            # Convertir a DataFrame
            datos_df = pd.DataFrame(datos_empresa, columns=['fecha', 'monto'])
            
            if datos_df.empty:
                return jsonify({'error': f'No hay datos para la empresa con ID {empresa_id}'}), 404
            
            # Preprocesar los datos
            montos = datos_df['monto'].values
            if len(montos) < 3:  # Se necesitan al menos 3 datos para predecir
                return jsonify({'error': 'La empresa no tiene suficientes datos históricos'}), 400
            
            X, scaler = preprocesar_datos(montos)
            
            # Realizar predicción
            ultima_secuencia = X[-1].reshape((1, X.shape[1], 1))
            predicciones = []
            
            for _ in range(pasos_futuros):
                pred = modelo.predict(ultima_secuencia, verbose=0)
                predicciones.append(pred[0, 0])
                # Actualizar secuencia
                ultima_secuencia = np.roll(ultima_secuencia, -1)
                ultima_secuencia[0, -1, 0] = pred[0, 0]
            
            # Invertir la normalización
            predicciones = scaler.inverse_transform(np.array(predicciones).reshape(-1, 1))
            predicciones = predicciones.flatten().tolist()
            
            # Devolver respuesta
            return jsonify({
                'empresa_id': empresa_id,
                'predicciones': predicciones
            })
        
        finally:
            # Devolver la conexión al pool
            conexion_pool.putconn(conn)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Prueba de conexión
@app.route('/', methods=['GET'])
def home():
    return jsonify({'mensaje': 'API de predicciones está funcionando correctamente'})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)