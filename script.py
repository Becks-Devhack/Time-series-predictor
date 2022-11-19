import tensorflow as tf
from flask import request, jsonify, Response, Flask
import pandas as pd 
import json

stress_classifier = tf.keras.models.load_model('stress_classifier')
app = Flask(__name__)

@app.route("/stress_level", methods=["GET"])
def get_stress_level():
    global stress_classifier
    payload = request.get_json(silent=True)
    print(request)
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
    global stress_classifier
    return jsonify({'prediction':False}), 200

if __name__ == '__main__':
    print(stress_classifier.summary())
    app.run('0.0.0.0', port=5000, debug=True)
