from flask_cors import CORS
from PIL import Image
import io
import uvicorn
import os
from io import BytesIO
from tempfile import SpooledTemporaryFile
from llm import ae, le
import tempfile
from openai import OpenAI
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from ultralytics import YOLO

from flask import Flask, request, jsonify, render_template
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import json
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def le(text):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="location_chat_history", return_messages=True)
    sys_prompt = """The user says: "{text}" Look for anything similar to the address or location in the user's 
            input and extract it. At most you should be able to at least extract the users location, then if it's well 
            detailed you can then extract the destination. Note: 1. There might be a case where the user didn't mention 
            the location, in that case, just return None. 2. The user's input is transcribed from the user's voice input. 
            So you will want to first of all analyze the text to extract the necessary information. Once you have the 
            location, return it (either one or both), return the location as a dictionary with these keys: "current location", "destination","""
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(sys_prompt),
                                               MessagesPlaceholder(variable_name="location_chat_history"),
                                               HumanMessagePromptTemplate.from_template("{text}")])
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

    memory.chat_memory.add_user_message(text)
    response = conversation.invoke({"text": text})
    return json.loads(response['text'])


# DB SCHEMA
Base = declarative_base()


class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True)
    user_input = Column(String)
    bot_response = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Interaction(id={self.id}, user_input='{self.user_input}', bot_response='{self.bot_response}', timestamp='{self.timestamp}')>"


client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)
text = None
emotion = None
yolov8_model = YOLO('yolov8m_imgsz614_map79_best.pt', verbose=False)

sqlite_db = "sqlite:///./patel_db.sqlite3"
engine = create_engine(sqlite_db)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


@app.route('/')
def home():
    return render_template('read4.html')


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    global emotion
    try:
        file = request.files['file']
        contents = file.read()
        im = Image.open(io.BytesIO(contents))
        yolo_res = yolov8_model.predict(im)
        pred = {}
        for j in yolo_res:
            class_name = j.names[int(j.boxes.cls[0])]
            pred.update({"pred_emotion": class_name})

        response = {"pred": "no detection" if not pred else pred}
        print(response)
        emotion = response
        return jsonify(response), 200
    except Exception as e:
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

        # Save interaction to the database
        interaction = Interaction(user_input=trans_text, bot_response=llm_result)
        session.add(interaction)
        session.commit()

        return jsonify({"transcribed_text": transcribed_text, "llm_output": llm_result})
    except Exception as e:
        # print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/get_directions', methods=['POST'])
def get_directions():
    data = request.json
    text = data.get('text', '')
    directions = le(text)
    return jsonify(directions)


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
