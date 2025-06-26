from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo y encoder
model = joblib.load('taco_sales.pkl')
encoder = joblib.load('encoder.pkl')

app.logger.debug('✅ Modelo y codificador cargados correctamente.')

# Mostrar las categorías aprendidas por el encoder
for name, cats in zip(['Taco Size', 'Taco Type'], encoder.categories_):
    app.logger.debug(f"✅ Categorías {name}: {list(cats)}")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener valores del formulario
        order_hour = int(request.form['order_hour'])
        taco_size = request.form['taco_size']
        taco_type = request.form['taco_type']
        toppings_count = float(request.form['toppings_count'])
        tip = float(request.form['tip'])
        weekend_order = int(request.form['weekend'])
        order_weekday = int(request.form['weekday'])

        app.logger.debug(f'Datos recibidos:\n{taco_size}, {taco_type}, {toppings_count}, {tip}, {weekend_order}, {order_weekday}')

        # Codificar datos categóricos
        categorical_input = pd.DataFrame([[taco_size, taco_type]],
                                         columns=['Taco Size', 'Taco Type'])
        encoded_cat = encoder.transform(categorical_input)

        # Crear DataFrame de entrada para el modelo
        final_input = pd.DataFrame(
         [[encoded_cat[0][0], encoded_cat[0][1],
             toppings_count, tip, weekend_order, order_hour, order_weekday]],
             columns=['Taco Size', 'Taco Type', 'Toppings Count',
             'Tip ($)', 'Weekend Order', 'Order Hour', 'Order Weekday']
)


        app.logger.debug(f'Tipos de datos:\n{final_input.dtypes}')
        app.logger.debug(f'📦 Datos para predicción:\n{final_input}')

        # Hacer predicción
        prediction = model.predict(final_input)
        precio = round(prediction[0], 2)

        app.logger.debug(f'💰 Precio estimado: {precio}')
        return jsonify({'precio_estimado': precio})

    except Exception as e:
        app.logger.error(f'🚨 Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
