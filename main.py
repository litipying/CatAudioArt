import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import io
import os
from PIL import Image
import librosa

# Set page config
st.set_page_config(page_title="Voice to Art Generator", layout="wide")

# Initialize session state variables
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'audio_file_path' not in st.session_state:
    st.session_state['audio_file_path'] = None

# Create audio folder if it doesn't exist
AUDIO_FOLDER = "audio_files"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

def record_audio(duration=10, sample_rate=44100):
    """Record audio for the specified duration"""
    recording = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1,
                      dtype='float32')
    sd.wait()
    return recording

def save_audio(audio_data, sample_rate=44100):
    """Save the recorded audio to a WAV file in the audio folder"""
    # Create a unique filename using timestamp
    filename = f"audio_{int(time.time())}.wav"
    # Create full filepath in the audio folder
    filepath = os.path.join(os.path.dirname(__file__), AUDIO_FOLDER, filename)
    write(filepath, sample_rate, audio_data)
    return filepath

def generate_art_from_prompt(prompt):
    """Generate art using Stability AI API"""
    # Initialize stability API client
    stability_api = client.StabilityInference(
        key='sk-1234567890',  # Replace with your actual API key
        verbose=True,
    )
    
    # Generate the image
    answers = stability_api.generate(
        prompt=prompt,
        seed=42,
        steps=30,
        cfg_scale=8.0,
        width=512,
        height=512,
        samples=1,
    )
    
    # Process the generated image
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                return img

def analyze_audio(audio_file_path):
    """Enhanced audio analysis for unique art generation"""
    # Load the audio file
    y, sr = librosa.load(audio_file_path)
    
    # Extract more detailed audio features
    # 1. Basic Features
    energy = np.mean(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 2. Spectral Features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
    # 3. Rhythm Features
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    rhythm_regularity = np.std(pulse)
    
    # 4. Harmonic Features
    harmonic, percussive = librosa.effects.hpss(y)
    harmonics_ratio = np.mean(harmonic) / (np.mean(percussive) + 1e-10)
    
    # 5. Calculate additional metrics
    dynamic_range = np.max(np.abs(y)) - np.min(np.abs(y))
    zero_crossings = np.mean(librosa.zero_crossings(y))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    
    # Generate base prompt based on primary characteristics
    art_style = ""
    color_scheme = ""
    composition = ""
    details = ""
    
    # Art Style selection based on harmonic content and rhythm
    if harmonics_ratio > 1.5:
        art_style = "flowing, organic, "
    elif harmonics_ratio < 0.5:
        art_style = "geometric, structured, "
    else:
        art_style = "balanced mix of organic and geometric shapes, "
        
    # Add rhythm influence
    if rhythm_regularity > 0.5:
        art_style += "with repeating patterns "
    else:
        art_style += "with spontaneous elements "
    
    # Color scheme based on spectral features and MFCCs
    mfcc_mean = np.mean(mfccs)
    if mfcc_mean > 0:
        if spectral_bandwidth > 1000:
            color_scheme = "vibrant complementary colors with "
        else:
            color_scheme = "harmonious analogous colors with "
    else:
        if spectral_bandwidth > 1000:
            color_scheme = "contrasting monochromatic scheme with "
        else:
            color_scheme = "subtle earth tones with "
            
    # Add temperature based on energy
    if energy > 0.1:
        color_scheme += "warm undertones, "
    else:
        color_scheme += "cool undertones, "
    
    # Composition based on tempo and dynamics
    if tempo > 120:
        if dynamic_range > 0.5:
            composition = "dynamic spiral composition "
        else:
            composition = "radiating circular patterns "
    elif tempo > 80:
        if dynamic_range > 0.5:
            composition = "diagonal flowing movements "
        else:
            composition = "gentle curved forms "
    else:
        if dynamic_range > 0.5:
            composition = "horizontal layered structure "
        else:
            composition = "minimal floating elements "
    
    # Details based on spectral features
    if spectral_rolloff.mean() > 0.5:
        details += "with intricate details in the foreground "
    else:
        details += "with subtle textures in the background "
        
    if zero_crossings > 0.1:
        details += "and sharp accents "
    else:
        details += "and smooth transitions "
    
    # Add unique elements based on specific frequency bands
    freq_bands = librosa.amplitude_to_db(
        np.abs(librosa.stft(y)), ref=np.max)
    if np.mean(freq_bands[:len(freq_bands)//3]) > -30:
        details += "featuring deep, grounding elements "
    if np.mean(freq_bands[len(freq_bands)//3:2*len(freq_bands)//3]) > -30:
        details += "with mid-range flowing movements "
    if np.mean(freq_bands[2*len(freq_bands)//3:]) > -30:
        details += "and crystalline highlights "
    
    # Combine all elements into final prompt
    prompt = f"Create a {art_style} artwork using {color_scheme} featuring a {composition} {details}"
    
    # Add artistic medium based on overall characteristics
    if energy > 0.1 and tempo > 100:
        prompt += "rendered in an expressive oil painting style"
    elif harmonics_ratio > 1.2:
        prompt += "rendered in a watercolor style with flowing pigments"
    else:
        prompt += "rendered in a detailed digital art style"
    
    return prompt

# Main app layout
st.title("Voice to Art Generator")
st.write("Record your voice and see it transformed into art!")

# Recording interface
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Recording", disabled=st.session_state['recording']):
        st.session_state['recording'] = True
        with st.spinner("Recording..."):
            audio_data = record_audio()
            st.session_state['audio_data'] = audio_data
            # Save the filepath in session state
            st.session_state['audio_file_path'] = save_audio(audio_data)
        st.session_state['recording'] = False
        st.success("Recording completed!")
        st.rerun()

    # Display audio player if we have a valid file path
    if st.session_state['audio_file_path'] is not None:
        try:
            st.audio(st.session_state['audio_file_path'])
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")

    # Add this after recording to debug file paths
    if st.session_state['audio_file_path'] is not None:
        st.write(f"Debug - Audio file path: {st.session_state['audio_file_path']}")
        st.write(f"Debug - File exists: {os.path.exists(st.session_state['audio_file_path'])}")

# Art generation interface
with col2:
    if st.button("Generate Art from Voice") and st.session_state['audio_file_path'] is not None:
        with st.spinner("Analyzing audio and generating art..."):
            prompt = analyze_audio(st.session_state['audio_file_path'])
            
            # Display the generated prompt in an expandable section
            with st.expander("View generated prompt"):
                st.write(prompt)
            
            generated_image = generate_art_from_prompt(prompt)
            st.image(generated_image, caption="Generated Art", use_column_width=True)

# Move cleanup to the end and only remove old files
def cleanup_old_files():
    """Clean up audio files older than 1 hour"""
    current_time = time.time()
    directory = os.path.join(os.path.dirname(__file__), AUDIO_FOLDER)
    for filename in os.listdir(directory):
        if filename.startswith("audio_") and filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            # Remove files older than 1 hour
            if current_time - os.path.getctime(filepath) > 3600:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error removing file {filepath}: {e}")

cleanup_old_files()
