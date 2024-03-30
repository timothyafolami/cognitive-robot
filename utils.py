from openai import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
# import streamlit as st
# import base64
import os
load_dotenv()


OpenAI_key = os.getenv('openai_api_key')

client = OpenAI(api_key=OpenAI_key)

# Define a function to get an answer from the chatbot
def get_answer(messages):
    # Create a system message to set the role and content of the AI chatbot
    system_message = [{"role": "system", "content": "You are a helpful AI chatbot that answers questions asked by the User."}]
    # Combine the system message with the user messages
    messages = system_message + messages
    # Send the messages to the OpenAI chat completions API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    # Return the content of the first choice from the response
    return response.choices[0].message.content

# Define a function to convert speech to text using the OpenAI audio transcriptions API
def speech_to_text(audio_data):
    # Open the audio file in binary mode
    with open(audio_data, "rb") as audio_file:
        # Create a transcription using the Whisper model
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    # Return the transcript
    return transcript

# Define a function to convert text to speech using the OpenAI audio speech API
def text_to_speech(input_text):
    # Create speech using the TTS model and the Nova voice
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    # Save the speech as an audio file
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    # Return the file path of the audio file
    return webm_file_path

