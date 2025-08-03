from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos entrenados
scaler = joblib.load('scaler_cluster.pkl')
encoder = joblib.load('encoder_cluster.pkl')
kmeans = joblib.load('kmeans_cluster_model.pkl')
pca = joblib.load('pca_2d_model.pkl')

app.logger.debug('✅ Modelos cargados correctamente (scaler, encoder, kmeans, pca)')

# Columnas esperadas
categoricas = ['producto_temporada', 'tipo_transaccion']
numericas = ['monto_total', 'mes_transaccion', 'vestido_en_oferta',
             'vestido_rating_promedio', 'vestido_review_count']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        # Recoger datos del formulario
        data = {
            'producto_temporada': request.form['producto_temporada'],
            'tipo_transaccion': request.form['tipo_transaccion'],
            'monto_total': float(request.form['monto_total']),
            'mes_transaccion': int(request.form['mes_transaccion']),
            'vestido_en_oferta': int(request.form['vestido_en_oferta']),
            'vestido_rating_promedio': float(request.form['vestido_rating_promedio']),
            'vestido_review_count': int(request.form['vestido_review_count'])
        }

        app.logger.debug(f"📥 Datos recibidos: {data}")

        # Crear DataFrame
        df_input = pd.DataFrame([data])

        # Escalar numéricos
        X_num = scaler.transform(df_input[numericas])

        # Codificar categóricos
        X_cat = encoder.transform(df_input[categoricas])

        # Unir
        X_final = np.concatenate([X_num, X_cat], axis=1)

        # Predecir clúster
        cluster = kmeans.predict(X_final)[0]

        # Reducir dimensión para visualización opcional
        X_pca = pca.transform(X_final)

        return jsonify({
            'cluster_asignado': int(cluster),
            'pca': {
                'PC1': float(X_pca[0][0]),
                'PC2': float(X_pca[0][1])
            }
        })

    except Exception as e:
        app.logger.error(f'🚨 Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
