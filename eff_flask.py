from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import uvicorn
import os
from io import BytesIO
from tempfile import SpooledTemporaryFile
from llm import ae
import tempfile
from openai import OpenAI


client = OpenAI(api_key="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")

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model('model_effNETb7.h5', compile=False)
emotion_results = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

text = None
emotion = None

@app.route('/')
def home():
    return render_template('real3.html')


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    global emotion
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
            roi = cv2.resize(fc, (224, 224))
            roi = tf.expand_dims(roi, 0)
            y_pred = emotion_model.predict(roi)
            top_two_indices = np.argsort(y_pred)[0, -2:]
            top_two_predictions = y_pred[0, top_two_indices]
            for i, index in enumerate(top_two_indices):
                emotion = emotion_results[index]
                percentage = round((top_two_predictions[i] * 100), 3)
                # print(f"{percentage:.2f}% {emotion}")
                pred.update({percentage: emotion})

        response = {"pred": "no detection" if not pred else pred}
        emotion = response

        print(response)
        return jsonify(response), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


trans_text = ""
@app.route("/audio2text", methods=["POST"])
def audio():
    global trans_text
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
        # print(transcription.text)
        transcribed_text = transcription.text
        trans_text = transcribed_text

        llm_result = process_text(trans_text)

        return jsonify({"transcribed_text": transcribed_text, "llm_output": llm_result})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


# @app.route('/process_llm_text', methods=['POST'])
def process_text(trans_text):
    # global trans_text
    global emotion
    # data = request.get_json()
    # input_text = data['input_text']
    input_text = trans_text + str(emotion)
    if input_text == "":
        input_text = "what's your purpose?"
    print('YOU: ', input_text)
    agent_executor = ae()
    result = agent_executor.invoke({"input": input_text})
    print({'NAVI': result['output']})
    # return jsonify({'output': result['output']})
    return result['output']


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
