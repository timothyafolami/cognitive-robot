# Import necessary libraries
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import uvicorn


# Create Flask app
app = Flask(__name__)
CORS(app)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load emotion detection model
model_json_file = 'model.json'
model_weights_file = 'model_weights.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)

# Define route for emotion prediction
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        # Read image file from request
        file = request.files['file']
        contents = file.read()
        arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        pred = ""

        # Convert image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        for (x, y, w, h) in faces:
            fc = gray[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))

            # Predict emotion for the face
            pred = loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])
            text_idx = np.argmax(pred)
            text_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
            text = text_list[text_idx]
            pred = text

        # Prepare response
        response = {"pred": "no detection" if not pred else pred}
        print(response)
        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
