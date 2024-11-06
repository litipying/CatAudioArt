# Voice-to-Art Generator

A Streamlit application that records voice input and generates artistic images based on audio characteristics. The app analyzes various audio features such as pitch, tempo, and energy to create unique artistic representations.

## Features

- Voice recording capability
- Real-time audio playback
- Audio analysis using librosa
- AI-powered art generation using Stability AI
- Automatic cleanup of temporary audio files

## Prerequisites

- Python 3.8 or higher
- A Stability AI API key (Get it from [stability.ai](https://stability.ai))

## Installation

1. Clone the repository:
```
git clone <your-repository-url>
cd voice-to-art-generator
```

2. Create and activate a virtual environment:
```
For macOS/Linux
python -m venv venv
source venv/bin/activate
For Windows
python -m venv venv
.\venv\Scripts\activate 
```

3. Install required packages:
```
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Stability AI API key:
```
STABILITY_AI_API_KEY=<your-stability-ai-api-key>
```

## Project Structure
voice-to-art-generator/
├── main.py
├── .env
├── requirements.txt
├── README.md
├── audio_files/ # Temporary audio storage
└── venv/ # Virtual environment

## Running the Application

1. Make sure your virtual environment is activated
2. Run the Streamlit app:
```
streamlit run main.py
```

3. Open your web browser and go to `http://localhost:8501`

## How to Use

1. Click the "Start Recording" button to record your voice
2. The recording will automatically stop after 5 seconds
3. You can play back your recording using the audio player
4. Click "Generate Art from Voice" to create an AI-generated image based on your voice characteristics
5. The generated image will be displayed on the right side of the screen

## Audio Analysis Features

The application analyzes several characteristics of your voice:
- Volume/Energy
- Pitch/Frequency
- Tempo
- Spectral contrast

These characteristics influence different aspects of the generated art:
- Loud sounds → Bold, large shapes
- High pitch → Bright colors
- Fast tempo → Dynamic patterns
- High contrast → Textured surfaces

## Requirements

Create a `requirements.txt` file with these dependencies:
```
streamlit
sounddevice
scipy
stability-sdk
Pillow
librosa
```