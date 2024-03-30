from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import uvicorn
import os
from io import BytesIO
from tempfile import SpooledTemporaryFile

from openai import OpenAI

client = OpenAI(api_key="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")

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model('model_mobNET.h5', compile=False)
emotion_results = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        file = request.files['file']
        contents = file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        pred = {}
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = frame[y:y + h, x:x + w]
            roi = cv2.resize(fc, (250, 250))
            roi = tf.expand_dims(roi, 0)
            y_pred = emotion_model.predict(roi)
            top_two_indices = np.argsort(y_pred)[0, -2:]
            top_two_predictions = y_pred[0, top_two_indices]
            for i, index in enumerate(top_two_indices):
                emotion = emotion_results[index]
                percentage = round((top_two_predictions[i] * 100), 3)
                # print(f"{percentage:.2f}% {emotion}")
                pred.update({percentage:emotion})
            
        
        response = {"pred": "no detection" if not pred else pred}
        print(response)
        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
        
import tempfile
        
@app.route("/audio2speech", methods=["POST"])
def audio():
    try:
        file = request.files['file']
        _, image_extension = os.path.splitext(file.filename)
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=image_extension)
        file.save(temp_image_path.name)
        
#         audio_file_path = os.path.join(os.getcwd(), 'uploads', 'recording.ogg')
#         os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)
#         file.save(audio_file_path)
        
        print(temp_image_path.name)
        
        audio_file = open(temp_image_path.name, 'rb')
        # print(audio_file)
        transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcription.text
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

