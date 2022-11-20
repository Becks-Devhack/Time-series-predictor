import tensorflow as tf
from flask import request, jsonify, Response, Flask
import pandas as pd 
from statsmodels.tsa.statespace.sarimax import SARIMAX

stress_classifier = tf.keras.models.load_model('src/models/stress_classifier')
app = Flask(__name__)

@app.route("/data", methods=["GET"])
def get_stress_level():
    global stress_classifier
    payload = request.get_json(silent=True)
    if not payload:
        # Error handling
        return Response(status=400)
    
    br = payload['breathing_rate']
    os = []
    sh = payload['sleep_hrs']
    hr = payload['heart_rate']
    
    max_l = max([len(br), len(sh), len(hr)])
    br = [br[i] if i < len(br) else 25 for i in range(max_l)]
    os = [os[i] if i < len(os) else 95 for i in range(max_l)]
    sh = [sh[i] if i < len(sh) else 6 for i in range(max_l)]
    hr = [hr[i] if i < len(hr) else 75 for i in range(max_l)]
    
    data = pd.DataFrame({'breathing_rate':br, 'oxygen_sat':os, 'sleep_hrs':sh, 'heart_rate':hr})
    print(data.shape)
    classification = stress_classifier.predict(data)
    print(classification.tolist())
    stress_level = [max(range(len(x)), key=x.__getitem__) for x in classification.tolist()]
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
