# AI Detector & Humanizer Pro

A production-ready AI content detector and humanizer built with FastAPI, Hugging Face Transformers, and Groq (Llama 3).

## Features
- **AI Detection:** Uses a RoBERTa-based classifier to estimate the probability of text being AI-generated.
- **AI Humanizer:** Leverages Llama 3 (via Groq) to rewrite text with natural "burstiness" and flow, bypassing AI detectors.
- **Verification Loop:** The humanizer automatically checks its output against the detector and regenerates if the score remains high.

## Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher.
- A Groq API Key (Get it at [console.groq.com](https://console.groq.com/)).

### 2. Installation
1. Clone this repository or download the source code.
2. Install the required Python libraries:
   ```bash
   pip install fastapi uvicorn transformers torch groq python-dotenv
   ```

### 3. Configuration
1. Open the `.env` file in the root directory.
2. Replace the placeholder with your actual Groq API Key:
   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

### 4. Running the Project
1. Start the FastAPI server:
   ```bash
   python main.py
   ```
2. Wait for the model to load. The first run will download approximately 500MB of model weights from Hugging Face.
3. Once you see `INFO: Uvicorn running on http://127.0.0.1:8000`, the server is ready.

### 5. Accessing the UI
1. Open `index.html` directly in your web browser.
2. Alternatively, navigate to `http://127.0.0.1:8000/` in your browser.

## Project Structure
- `main.py`: FastAPI backend handling detection and humanization logic.
- `index.html`: Simple Tailwind-based frontend.
- `.env`: Environment variables (API Keys).
- `.gitignore`: Standard Python and environment exclusion rules.
