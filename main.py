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
import torch
import math
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel, PreTrainedModel
import torch.nn as nn
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

# --- ENSEMBLE DETECTOR CONFIG ---
# 1. DeBERTa-v3 Classifier (Industry standard for precision)
# Using AutoModelForSequenceClassification is more robust against Transformers version changes
print("Loading DeBERTa-v3 Classifier...")
classifier_model_id = "desklib/ai-text-detector-v1.01" 
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)

try:
    # Attempt to load using standard sequence classification class (works for most modern HF models)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_id)
except Exception as e:
    print(f"Standard loading failed, using robust custom architecture fallback...")
    # Fallback to a custom class that handles internal weight tying correctly for newer library versions
    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig
        def __init__(self, config):
            super().__init__(config)
            # The checkpoint uses 'model' as the attribute name for the base transformer
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            # post_init() handles modern weight initialization and tying
            self.post_init()
        
        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            # Accept **kwargs to handle 'token_type_ids' passed by the tokenizer
            outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
            last_hidden_state = outputs[0]
            
            # Global Average Pooling (Mean Pooling)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            
            logits = self.classifier(pooled_output)
            return type('Outputs', (object,), {'logits': logits})()

    classifier_model = DesklibAIDetectionModel.from_pretrained(classifier_model_id)

classifier_model.eval()
if torch.cuda.is_available():
    classifier_model = classifier_model.to("cuda")

# 2. Perplexity and Binoculars-proxy Engine (GPT-2 for efficiency)
print("Loading Perplexity & Entropy Engine (GPT-2)...")
ppl_model_id = "gpt2"
ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_id)
ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_id)
if torch.cuda.is_available():
    ppl_model = ppl_model.to("cuda")

class TextRequest(BaseModel):
    text: str

def count_tokens(text: str):
    """Accurately count tokens using tiktoken (cl100k_base is used as a proxy for Llama 3)."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        return len(text) // 4 # Fallback to rough estimate

def get_perplexity(text: str):
    """Calculates perplexity of the text using GPT-2. Human text = High Perplexity."""
    if not text.strip():
        return 0
    try:
        inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = ppl_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        # Normalize: Lower perplexity -> Higher AI probability
        norm_score = max(0, min(100, (100 - (perplexity / 1.5))))
        return norm_score
    except:
        return 50

def get_binoculars_score(text: str):
    """
    Zero-shot detection proxy. 
    Uses token distribution entropy as a measure of predictability.
    AI text tends to have lower entropy (highly predictable).
    """
    try:
        inputs = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = ppl_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            
        # Lower entropy -> higher predictability (AI-like)
        norm_score = max(0, min(100, (8 - entropy) * 12.5)) 
        return norm_score
    except:
        return 50

def get_ai_score(text: str):
    """
    Returns the ensemble probability of the text being AI-generated.
    Weighting: 40% DeBERTa, 40% Binoculars (Proxy), 20% Perplexity
    """
    # 1. DeBERTa Neural Classification (40%)
    inputs = classifier_tokenizer(text[:1500], return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        # Handle both dict-like and attribute-like access for maximum compatibility
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs[0] # Fallback for tuple returns
            
        # Convert single logit to probability
        deberta_score = torch.sigmoid(logits).item() * 100


    # 2. Binoculars Predictability Analysis (40%)
    binoculars_score = get_binoculars_score(text[:1000])

    # 3. Statistical Perplexity Check (20%)
    perplexity_score = get_perplexity(text[:1000])

    # Final Weighted Calculation
    final_score = (deberta_score * 0.40) + (binoculars_score * 0.40) + (perplexity_score * 0.20)
    
    return round(final_score, 2)

def chunk_text_by_tokens(text: str, max_tokens: int):
    """Splits text into chunks based on token counts."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

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
        yield json.dumps({"log": "Initializing Ensemble Detection Engine...", "log_type": "process"}) + "\n"
        
        # Calculate sub-scores for detailed logging
        ppl = get_perplexity(text[:1000])
        yield json.dumps({"log": f"Statistical Perplexity: {round(ppl, 1)}% AI signals.", "log_type": "info"}) + "\n"
        
        bino = get_binoculars_score(text[:1000])
        yield json.dumps({"log": f"Binoculars Entropy: {round(bino, 1)}% predictability.", "log_type": "info"}) + "\n"
        
        yield json.dumps({"log": "Running DeBERTa-v3 neural classifier...", "log_type": "process"}) + "\n"
        score = get_ai_score(text)
        
        yield json.dumps({"log": f"Final Weighted Confidence: {score}%", "log_type": "info"}) + "\n"
        
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