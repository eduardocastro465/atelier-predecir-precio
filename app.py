from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos entrenados
modelo_renta = joblib.load('modelo_renta.pkl')
modelo_venta = joblib.load('modelo_venta.pkl')
encoders = joblib.load('LabelEncoders.pkl')  # Diccionario de encoders por columna

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
    "cintura",
]

@app.route('/predecir_precio', methods=['POST'])
def predecir_precio():
    try:
        data = request.json
        app.logger.debug(f"📥 JSON recibido: {data}")

        # Validar que lleguen todas las columnas necesarias
        for col in features + ["opcionesTipoTransaccion"]:
            if col not in data:
                raise ValueError(f"Falta el campo requerido: {col}")

        # Convertir venta/renta a número
        mapping_tipo = {"venta": 0, "renta": 1}
        tipo_valor = str(data["opcionesTipoTransaccion"]).strip().lower()
        if tipo_valor not in mapping_tipo:
            raise ValueError(f"Valor inválido en opcionesTipoTransaccion: {tipo_valor}")
        data["opcionesTipoTransaccion"] = mapping_tipo[tipo_valor]

        # Crear DataFrame
        df_input = pd.DataFrame([data])

        # Normalizar columnas que venían como listas
        if "talla" in df_input.columns and isinstance(df_input.loc[0, "talla"], list):
            df_input.loc[0, "talla"] = ", ".join(df_input.loc[0, "talla"])
        if "color" in df_input.columns and isinstance(df_input.loc[0, "color"], list):
            df_input.loc[0, "color"] = ", ".join(df_input.loc[0, "color"])

        # Codificar usando los encoders guardados
        for col in df_input.select_dtypes(include=["object", "bool"]).columns:
            if col in encoders:
                le = encoders[col]
                clases = list(le.classes_)

                # Agregar un marcador para valores desconocidos si no existe
                if "__desconocido__" not in clases:
                    clases.append("__desconocido__")
                    le.classes_ = np.array(clases)

                # Reemplazar valores no vistos por "__desconocido__"
                df_input[col] = df_input[col].apply(
                lambda x: x if x in le.classes_ else "__desconocido__"
                )
                df_input[col] = le.transform(df_input[col].astype(str))

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
