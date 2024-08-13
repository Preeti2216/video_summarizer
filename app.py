import streamlit as st
import tempfile
import os
from moviepy.editor import VideoFileClip
import yt_dlp
from gtts import gTTS
from translatepy import Translator
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
 
# Configure API key for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
 
# Initialize the translator
translator = Translator()
 
# Initialize the Groq client with the API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
# Function to download video from URL using yt-dlp
def download_video(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': tempfile.mktemp(suffix='.webm'),
        'merge_output_format': 'webm',
        'noplaylist': True,
    }
   
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = info_dict.get('filepath', ydl.prepare_filename(info_dict))
   
    return video_path
 
def extract_audio(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
 
    video = VideoFileClip(video_path)
    audio = video.audio
 
    # Determine the audio file path based on the video format
    temp_audio_path = video_path
    if video_path.endswith(".webm"):
        temp_audio_path = video_path.replace(".webm", ".mp3")
    elif video_path.endswith(".mp4"):
        temp_audio_path = video_path.replace(".mp4", ".mp3")
    else:
        temp_audio_path = video_path + ".mp3"  # Default to appending ".mp3" if format is unknown
 
    # Extract and save the audio
    audio.write_audiofile(temp_audio_path)
 
    # Close the video and audio objects
    video.close()
    audio.close()
 
    return temp_audio_path
 
# Function to transcribe audio to text using Groq
def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                prompt="Specify context or spelling",  # Optional
                response_format="json",  # Optional
                language="en",  # Optional
                temperature=0.0  # Optional
            )
        return transcription.text
    except Exception as e:
        return f"An error occurred during transcription: {e}"
 
# Define the prompt for summarization
def create_prompt(length):
    return f"""You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary within {length} words. Please provide the summary of the text given here: """
 
def summarize_text(text, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + text)
        return response.text
    except Exception as e:
        return f"An error occurred during summarization: {e}"
 
# Function to translate text
def translate_text(text, target_language):
    try:
        translated = translator.translate(text, target_language)
        return str(translated)  # Ensure the result is a string
    except Exception as e:
        return f"An error occurred during translation: {e}"
 
# Function to generate audio from text using gTTS
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        audio_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"An error occurred during text-to-speech: {e}"
 
def main():
    st.title("Video Summarizer and Translator")
 
    # Slider for summary length
    summary_length = st.slider(
        "Select length of the summary (in words):",
        min_value=50, max_value=500, value=250, step=10
    )
 
    # Upload or URL input
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    video_url = st.text_input("Or enter a video URL")
 
    video_path = None
    audio_path = None
 
    if uploaded_file is not None:
        # Handle file upload
        video_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(video_path)
        st.write("Processing your video...")
 
    elif video_url:
        # Handle URL input
        st.write("Downloading video...")
        try:
            video_path = download_video(video_url)
            st.video(video_path)
            st.write("Video downloaded successfully.")
        except Exception as e:
            st.write(f"An error occurred during video download: {e}")
 
    if video_path:
        try:
            # Extract and transcribe audio
            audio_path = extract_audio(video_path)
            if audio_path:
                transcript = transcribe_audio(audio_path)
 
                # Summarize text
                st.write("Summarizing text...")
                prompt = create_prompt(summary_length)  # Create prompt with user-specified length
                summary = summarize_text(transcript, prompt)
                st.write("Summary:")
                st.write(summary)
 
                # Translate text
                language_options = {
                    "en": "English", "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "ta": "Tamil", "kn": "Kannada",
                    "ur": "Urdu", "gu": "Gujarati", "ml": "Malayalam", "or": "Odia", "pa": "Punjabi", "as": "Assamese",
                    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian", "pt": "Portuguese", "zh-cn": "Chinese (Simplified)",
                    "ja": "Japanese", "ko": "Korean", "ar": "Arabic"
                }
                target_language = st.selectbox("Select language to translate the summary", list(language_options.values()))
                target_language_code = [code for code, name in language_options.items() if name == target_language][0]
               
                translated_summary = translate_text(summary, target_language_code)
                st.write(f"Translated Summary ({target_language}):")
                st.write(translated_summary)
 
                # Generate and play audio
                st.write("Generating audio from summary...")
                audio_summary_path = text_to_speech(translated_summary, lang=target_language_code)
                if isinstance(audio_summary_path, str) and os.path.exists(audio_summary_path):
                    st.audio(audio_summary_path, format='audio/mp3')
                else:
                    st.write(f"Failed to generate audio or invalid path: {audio_summary_path}")
 
            else:
                st.write("Failed to extract audio from the video.")
 
        except Exception as e:
            st.write(f"An error occurred during processing: {e}")
        finally:
            # Clean up temporary files
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except PermissionError as e:
                    st.write(f"Failed to delete video file: {e}")
 
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except PermissionError as e:
                    st.write(f"Failed to delete audio file: {e}")
 
if __name__ == "__main__":
    main()
