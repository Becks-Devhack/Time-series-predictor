import tensorflow as tf
from flask import request, jsonify, Response, Flask
import pandas as pd 
import json

stress_classifier = tf.keras.models.load_model('stress_classifier')
app = Flask(__name__)

movies = {}
max_id = 1

@app.route("/stress_level", methods=["GET"])
def get_stress_level():
    global stress_classifier
    payload = request.get_json(silent=True)
    print(request)
    if not payload:
        # Error handling
        return Response(status=400)
    br = payload['breathing_rate']
    bt = payload['body_temp']
    os = payload.get('oxygen_sat', 90)
    sh = payload['sleep_hrs']
    hr = payload['heart_rate']
    
    data = pd.DataFrame([[0, br,bt,os,sh, hr]], columns=['', 'breathing_rate', 'body_temp', 'oxygen_sat', 'sleep_hrs', 'heart_rate'])
    stress_level = stress_classifier.predict(data).argmax().tolist()
    return jsonify({'stress level':stress_level}), 200

@app.route("/vacantion_prediction", methods=["GET"])
def get_vacantion_prediction():
    global stress_classifier
    return jsonify([{'id':k, 'nume':v} for k,v in movies.items()]), 200

# @app.route('/movies', methods=["POST"])
# def add_movie():
#     global movies
#     global max_id
#     payload = request.get_json(silent=True)
#     if not payload:
#         # Error handling
#         return Response(status=400)
#     movies[max_id] = payload['nume']
#     max_id += 1
#     return Response(status=201)

if __name__ == '__main__':
    print(stress_classifier.summary())
    app.run('0.0.0.0', port=5000, debug=True)