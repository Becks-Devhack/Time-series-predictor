import tensorflow as tf
from flask import request, jsonify, Response, Flask
import pandas as pd 
from statsmodels.tsa.statespace.sarimax import SARIMAX

stress_classifier = tf.keras.models.load_model('models/stress_classifier')
app = Flask(__name__)

@app.route("/stress_level", methods=["GET"])
def get_stress_level():
    global stress_classifier
    payload = request.get_json(silent=True)
    if not payload:
        # Error handling
        return Response(status=400)
    
    br = payload['breathing_rate']
    os = payload.get('oxygen_sat', 90)
    if os == "":
        os = 90
    sh = payload['sleep_hrs']
    hr = payload['heart_rate']
    
    data = pd.DataFrame([[br, os, sh, hr]], columns=['breathing_rate', 'oxygen_sat', 'sleep_hrs', 'heart_rate'])
    classification = stress_classifier.predict(data)
    print(classification)
    stress_level = classification.argmax().tolist()
    return jsonify({'stress level':stress_level}), 200

@app.route("/vacantion_prediction", methods=["GET"])
def get_vacantion_prediction():
    payload = request.get_json(silent=True)
    if not payload:
        # Error handling
        return Response(status=400)
    stress = payload["time_series"]
    data = {"Stress":stress}
    parsed_data = pd.DataFrame(data, dtype=float)

    SARIMAXmodel = SARIMAX(parsed_data["Stress"], order = (1, 0, 1))
    SARIMAXmodel = SARIMAXmodel.fit()

    y_pred = SARIMAXmodel.get_forecast(10) # Magic number
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    
    vacantion = False
    if y_pred_df["Predictions"].values[0] - y_pred_df["Predictions"].values[-1] > 0.35 and y_pred_df["Predictions"].values[-1] >=3:
        vacantion = True
    
    return jsonify({'prediction':vacantion}), 200

if __name__ == '__main__':
    print(stress_classifier.summary())
    app.run('0.0.0.0', port=80, debug=True)
