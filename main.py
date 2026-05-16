import sys
import io

# Redirect stdout and stderr to prevent crashes in windowed/noconsole mode
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json
from transformers import pipeline
from groq import Groq
import os
import io
import pypdf
import tiktoken
import time
from dotenv import load_dotenv

import sys

# Determine the base path for bundled files
if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file in the current working directory (not bundled)
ENV_PATH = os.path.join(os.getcwd(), ".env")
load_dotenv(ENV_PATH)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
CONTEXT_WINDOW = 128000 
CHUNK_TOKEN_LIMIT = 4000 

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)

client = get_groq_client()

app = FastAPI(title="AI Detector & Humanizer API")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the AI Detector Model (RoBERTa-base-openai-detector)
print("Loading AI Detector model... please wait.")
detector_pipe = pipeline("text-classification", model="roberta-base-openai-detector")

class TextRequest(BaseModel):
    text: str

def count_tokens(text: str):
    """Accurately count tokens using tiktoken (cl100k_base is used as a proxy for Llama 3)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text) // 4 # Fallback to rough estimate

def chunk_text_by_tokens(text: str, max_tokens: int):
    """Splits text into chunks based on token counts."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

def get_ai_score(text: str):
    """Returns the probability of the text being AI-generated."""
    # RoBERTa has a 512 token limit. We use the first ~1000 chars as a safe proxy.
    results = detector_pipe(text[:1000])
    score = 0
    for res in results:
        if res['label'] == 'Fake':
            score = res['score']
        else:
            score = 1 - res['score']
    return round(score * 100, 2)

def humanize_logic(text: str):
    """The 'Secret Sauce' Humanizer Loop with automatic chunking for rate-limit safety."""
    system_prompt = (
        "Rewrite the following text to make it indistinguishable from human writing. "
        "Use varying sentence structures (burstiness), natural idioms, and a conversational flow. "
        "Ensure it passes AI detection while maintaining the original meaning and professional quality. "
        "Avoid over-used AI words like 'delve', 'meticulous', and 'comprehensive'."
    )
    
    token_count = count_tokens(text)
    
    if token_count <= CHUNK_TOKEN_LIMIT:
        return _process_chunk(text, system_prompt)
    
    print(f"Large text detected ({token_count} tokens). Processing in chunks...")
    chunks = chunk_text_by_tokens(text, CHUNK_TOKEN_LIMIT)
    humanized_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"Humanizing chunk {i+1}/{len(chunks)}...")
        humanized_chunks.append(_process_chunk(chunk, system_prompt))
        # Small sleep to help avoid hitting RPM (Requests Per Minute) limits
        if i < len(chunks) - 1:
            time.sleep(1.5)
            
    return "\n\n".join(humanized_chunks)

def _process_chunk(chunk_text: str, system_prompt: str):
    """Internal helper to humanize individual segments."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text}
            ],
            temperature=0.8,
        )
        humanized = completion.choices[0].message.content
        
        # Quick verification/re-pass if score is still high
        if get_ai_score(humanized) > 30:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "The previous attempt still looks like AI. Rewrite more creatively with varied sentence lengths."},
                    {"role": "user", "content": humanized}
                ],
                temperature=0.9,
            )
            humanized = completion.choices[0].message.content
        return humanized
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower():
            raise HTTPException(status_code=429, detail="API Rate Limit Exceeded. Please try a smaller text or wait a minute.")
        raise e

async def stream_detection(text: str):
    try:
        yield json.dumps({"log": "Analyzing linguistic patterns...", "log_type": "process"}) + "\n"
        time.sleep(0.5)
        yield json.dumps({"log": "Running RoBERTa classifier...", "log_type": "process"}) + "\n"
        score = get_ai_score(text)
        yield json.dumps({"log": f"Calculated AI Probability: {score}%", "log_type": "info"}) + "\n"
        
        status = "AI" if score > 50 else "Human"
        yield json.dumps({
            "final_result": {"ai_probability": f"{score}%", "status": status}
        }) + "\n"
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

@app.post("/detect")
async def detect(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    return StreamingResponse(stream_detection(request.text), media_type="application/x-ndjson")

async def stream_humanization(text: str):
    try:
        system_prompt = (
            "Rewrite the following text to make it indistinguishable from human writing. "
            "Use varying sentence structures (burstiness), natural idioms, and a conversational flow. "
            "Ensure it passes AI detection while maintaining the original meaning and professional quality. "
            "Avoid over-used AI words like 'delve', 'meticulous', and 'comprehensive'."
        )
        
        token_count = count_tokens(text)
        yield json.dumps({"log": f"Input analysis: {token_count} tokens detected.", "log_type": "info"}) + "\n"
        
        if token_count <= CHUNK_TOKEN_LIMIT:
            yield json.dumps({"log": "Processing single segment...", "log_type": "process"}) + "\n"
            humanized = _process_chunk(text, system_prompt)
            yield json.dumps({"log": "Applying secondary verification pass...", "log_type": "process"}) + "\n"
        else:
            yield json.dumps({"log": f"Large text detected. Splitting into chunks...", "log_type": "warning"}) + "\n"
            chunks = chunk_text_by_tokens(text, CHUNK_TOKEN_LIMIT)
            humanized_chunks = []
            
            for i, chunk in enumerate(chunks):
                yield json.dumps({"log": f"Humanizing chunk {i+1} of {len(chunks)}...", "log_type": "process"}) + "\n"
                humanized_chunks.append(_process_chunk(chunk, system_prompt))
                if i < len(chunks) - 1:
                    yield json.dumps({"log": "Cooling down for rate limit safety...", "log_type": "info"}) + "\n"
                    time.sleep(1.5)
            humanized = "\n\n".join(humanized_chunks)

        yield json.dumps({"log": "Calculating final AI score...", "log_type": "process"}) + "\n"
        new_score = get_ai_score(humanized)
        
        yield json.dumps({
            "final_result": {
                "original_text": text,
                "humanized_text": humanized,
                "new_ai_score": f"{new_score}%"
            }
        }) + "\n"
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg.lower():
            error_msg = "Groq API Rate Limit Hit. Please wait a moment or use shorter text."
        yield json.dumps({"error": error_msg}) + "\n"

@app.post("/humanize")
async def humanize(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    return StreamingResponse(stream_humanization(request.text), media_type="application/x-ndjson")

@app.post("/count-tokens")
async def get_token_count(request: TextRequest):
    return {"tokens": count_tokens(request.text)}

@app.get("/limits")
async def get_limits():
    """Returns the model limits for the frontend to display."""
    return {
        "model": MODEL_NAME,
        "context_window": CONTEXT_WINDOW,
        "chunk_size": CHUNK_TOKEN_LIMIT,
        "explanation": "Groq's free tier has Rate Limits (TPM/RPM). Text exceeding the chunk size will be automatically split."
    }

@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """Extracts text content from an uploaded PDF file."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")
    
    try:
        pdf_content = await file.read()
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from this PDF (it might be an image-only scan).")
            
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

@app.post("/save-api-key")
async def save_api_key(request: dict):
    key = request.get("key")
    if not key:
        raise HTTPException(status_code=400, detail="Key is required")
    
    with open(ENV_PATH, "w") as f:
        f.write(f"GROQ_API_KEY={key}")
    
    # Reload env and client
    load_dotenv(ENV_PATH, override=True)
    global client
    client = get_groq_client()
    return {"status": "success"}

@app.get("/check-config")
async def check_config():
    return {
        "has_api_key": os.getenv("GROQ_API_KEY") is not None,
        "env_location": ENV_PATH
    }

@app.get("/")
async def serve_index():
    index_path = os.path.join(BASE_PATH, "index.html")
    return FileResponse(index_path)

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open("http://127.0.0.1:8000")

    # Start browser after a short delay to ensure server is up
    Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)