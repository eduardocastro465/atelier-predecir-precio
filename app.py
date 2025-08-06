from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos entrenados
modelo_renta = joblib.load('modelo_renta.pkl')
modelo_venta = joblib.load('modelo_venta.pkl')
encoder = joblib.load('LabelEncoder.pkl')  # Encoder entrenado en Jupyter

app.logger.debug('✅ Modelos cargados correctamente (renta, venta, encoder)')

# Columnas esperadas (mismas que en tu Jupyter)
features = [
    "tipoCola",
    "altura",
    "estilo",
    "talla",
    "tipoCuello",
    "color",
    "tipoHombro",
    "nombre",
    "descripcion"
]


@app.route('/predecir', methods=['POST'])
def predecir_precio():
    try:
        data = request.json
        app.logger.debug(f"📥 JSON recibido: {data}")

        # Validar que lleguen todas las columnas necesarias
        for col in features + ["opcionesTipoTransaccion"]:
            if col not in data:
                raise ValueError(f"Falta el campo requerido: {col}")

        # Convertir venta/renta a número como en Jupyter
        mapping_tipo = {"venta": 0, "renta": 1}
        tipo_valor = str(data["opcionesTipoTransaccion"]).strip().lower()
        if tipo_valor not in mapping_tipo:
            raise ValueError(f"Valor inválido en opcionesTipoTransaccion: {tipo_valor}")
        data["opcionesTipoTransaccion"] = mapping_tipo[tipo_valor]

        # Crear DataFrame
        df_input = pd.DataFrame([data])

        # Aplicar el mismo preprocesamiento que en Jupyter
        for col in df_input.select_dtypes(include=["object", "bool"]).columns:
            df_input[col] = encoder.fit_transform(df_input[col].astype(str))

        # Seleccionar modelo según tipo de transacción
        if data["opcionesTipoTransaccion"] == 1:
            modelo = modelo_renta
        else:
            modelo = modelo_venta

        # Predecir
        X_input = df_input[features]
        prediccion = modelo.predict(X_input)[0]

        return jsonify({
            "precio_estimado": float(prediccion),
            "tipo_transaccion": "renta" if data["opcionesTipoTransaccion"] == 1 else "venta"
        })

    except Exception as e:
        app.logger.error(f'🚨 Error: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)