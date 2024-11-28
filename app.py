import streamlit as st
import requests
from gtts import gTTS
from pydub import AudioSegment
from pydub.utils import mediainfo
from translate import Translator
from groq import Groq
from pydantic import BaseModel
from typing import Dict
import json
import os

# AssemblyAI API configuration
ASSEMBLYAI_API_KEY = st.secrets["ASSEMBLYAI_API_KEY"]
ASSEMBLYAI_ENDPOINT = "https://api.assemblyai.com/v2/transcript"

# Groq API configuration
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)


# Pydantic Model for JSON Schema
class RefinedTranscription(BaseModel):
    refined_transcription: str


# Function to preprocess audio (convert to required format)
def preprocess_audio(audio_data):
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_data)

    audio = AudioSegment.from_file("temp_audio.wav")
    audio = audio.set_channels(1)  # Ensure mono
    audio = audio.set_frame_rate(16000)  # Ensure 16kHz
    audio.export("processed_audio.wav", format="wav")

    with open("processed_audio.wav", "rb") as f:
        return f.read()


# Function to transcribe audio using AssemblyAI
def transcribe_audio(audio_file):
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    upload_url = "https://api.assemblyai.com/v2/upload"
    upload_response = requests.post(upload_url, headers=headers, data=audio_file)
    if upload_response.status_code != 200:
        st.error(f"Failed to upload audio. Status: {upload_response.status_code}")
        return None

    audio_url = upload_response.json().get("upload_url")
    if not audio_url:
        st.error("Upload URL not received from AssemblyAI.")
        return None

    transcript_request = {"audio_url": audio_url}
    transcript_response = requests.post(ASSEMBLYAI_ENDPOINT, json=transcript_request, headers=headers)
    if transcript_response.status_code != 200:
        st.error(f"Failed to request transcription. Status: {transcript_response.status_code}")
        return None

    transcript_id = transcript_response.json().get("id")
    if not transcript_id:
        st.error("Transcription ID not received.")
        return None

    transcript_status = "processing"
    while transcript_status not in ["completed", "failed"]:
        transcript_result = requests.get(f"{ASSEMBLYAI_ENDPOINT}/{transcript_id}", headers=headers)
        if transcript_result.status_code != 200:
            st.error(f"Failed to fetch transcription result. Status: {transcript_result.status_code}")
            return None

        transcript_status = transcript_result.json().get("status")
        if transcript_status == "completed":
            return transcript_result.json().get("text")
        elif transcript_status == "failed":
            st.error("Transcription failed.")
            return None

    return None


# Function to enhance transcription with Groq LLM in JSON Mode
def refine_transcription_with_groq(transcription: str) -> RefinedTranscription:
    try:
        schema = RefinedTranscription.model_json_schema()

        # Improved Prompt
        system_prompt = (
            "You are a medical transcription assistant. Your job is to check the provided transcription for "
            "any inaccuracies, particularly in medical terms. Follow these rules:\n"
            "1. If the transcription is accurate, return it exactly as provided, without any changes.\n"
            "2. If there are inaccuracies, correct them while retaining the original phrasing as much as possible.\n"
            "3. Do not add any additional content or fabricate information.\n"
            "Output the refined transcription in JSON format using the schema below:\n\n"
            f"{json.dumps(schema, indent=2)}"
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the transcription: {transcription}"}
            ],
            model="llama3-8b-8192",
            temperature=0,  # Deterministic output
            stream=False,
            response_format={"type": "json_object"}
        )

        # Log the raw response from the API
        raw_response = chat_completion.choices[0].message.content
        st.write("Raw Response from Groq:", raw_response)

        # Validate and return the refined transcription
        refined_transcription = RefinedTranscription.model_validate_json(raw_response)
        return refined_transcription
    except Exception as e:
        st.error(f"An error occurred during refinement with Groq: {e}")
        return None



# Function to translate text using translate-python
def translate_text(text, target_language):
    try:
        translator = Translator(to_lang=target_language)
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"An error occurred during translation: {e}")
        return None


# Function to convert text to speech using gTTS
def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"


# Main function
def main():
    st.title("Healthcare Translation Web App with Generative AI")

    input_choice = st.radio("Choose input method:", ["Upload audio file", "Record from microphone"])

    audio_data = None
    if input_choice == "Upload audio file":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
        if uploaded_file:
            audio_data = uploaded_file.read()
    elif input_choice == "Record from microphone":
        st.info("Click 'Record' to start recording and 'Stop' to end.")
        audio_input = st.audio_input("Record your message")
        if audio_input:
            audio_data = audio_input.getvalue()

    if audio_data:
        st.info("Processing audio...")
        processed_audio = preprocess_audio(audio_data)

        st.info("Transcribing audio...")
        transcription = transcribe_audio(processed_audio)
        if transcription:
            st.write("**Original Transcription:**")
            st.write(transcription)
            st.write("**Refined Transcription for Medical Terms through Generative AI:**")
            st.info("Refining transcription with Groq...")
            refined_transcription = refine_transcription_with_groq(transcription)
            if refined_transcription:
                st.write("**Refined Transcription:**")
                st.json(refined_transcription.dict())

                target_language = st.selectbox(
                    "Select target language",
                    ["es", "fr", "de", "zh", "en"],
                    index=0
                )

                st.info("Translating refined transcription...")
                translation = translate_text(refined_transcription.refined_transcription, target_language)
                if translation:
                    st.write(f"**Translation ({target_language}):**")
                    st.write(translation)

                    st.info("Converting translation to speech...")
                    audio_file = text_to_speech(translation, target_language)
                    audio = AudioSegment.from_mp3(audio_file)
                    audio.export("output.wav", format="wav")
                    st.audio("output.wav", format="audio/wav")
                    os.remove("output.mp3")
                    os.remove("output.wav")
            else:
                st.error("Refinement failed.")
        else:
            st.error("Transcription failed.")


if __name__ == "__main__":
    main()
