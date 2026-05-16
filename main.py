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
import re
import sqlite3
import hashlib
import asyncio
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
LOG_DATA_FOR_FINETUNING = True # Toggle for dataset collection
DATASET_FILE = os.path.join(os.getcwd(), "fine_tuning_dataset.jsonl")
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

# --- CACHE SETUP ---
CACHE_DB = os.path.join(os.getcwd(), "app_cache.db")

def init_db():
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cache 
                 (key TEXT PRIMARY KEY, value TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def get_cache(key_str: str):
    key = hashlib.sha256(key_str.encode()).hexdigest()
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("SELECT value FROM cache WHERE key=?", (key,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def log_for_finetuning(original: str, humanized: str, tone: str, permission: bool = True):
    """Logs successful humanizations for future fine-tuning dataset creation."""
    if not LOG_DATA_FOR_FINETUNING or not permission:
        return
    
    try:
        data = {
            "instruction": f"Rewrite the following text in a {tone} human-like tone.",
            "input": original,
            "output": humanized,
            "timestamp": time.time()
        }
        with open(DATASET_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error logging dataset: {e}")

def set_cache(key_str: str, value: dict):
    key = hashlib.sha256(key_str.encode()).hexdigest()
    conn = sqlite3.connect(CACHE_DB)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", 
              (key, json.dumps(value)))
    conn.commit()
    conn.close()

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
    tone: str = "balanced"
    allow_logging: bool = True

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

def get_burstiness_score(text: str):
    """Calculates sentence length variance (Burstiness). AI text is uniform; Humans use bursts."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 3:
        return 50
    
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    
    # High std_dev (> 12) -> Human (0 score). Low std_dev (< 3) -> AI (100 score).
    norm_score = max(0, min(100, 100 - (std_dev * 6)))
    return norm_score

def get_ai_fingerprint_score(text: str):
    """Checks for density of 'LLMisms' (Fingerprints)."""
    llmisms = ['delve', 'testament', 'comprehensive', 'meticulous', 'unlocking', 
               'pivotal', 'synergy', 'transformative', 'vibrant', 'embark', 
               'underscore', 'tailored', 'seamless', 'holistic', 'invaluable', 'notably']
    
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0
    
    hits = sum(1 for word in words if word in llmisms)
    density = (hits / len(words)) * 100
    
    # Approximately 1% density of these specific words is a strong AI indicator
    norm_score = max(0, min(100, density * 120))
    return norm_score

def get_ai_detailed_report(text: str):
    """
    Returns the ensemble probability and a detailed reasoning for the score.
    """
    # 1. DeBERTa Neural Classification (30%)
    inputs = classifier_tokenizer(text[:1500], return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs[0]
        deberta_score = torch.sigmoid(logits).item() * 100

    # 2. Binoculars Predictability Analysis (25%)
    binoculars_score = get_binoculars_score(text[:1000])

    # 3. Statistical Perplexity Check (15%)
    perplexity_score = get_perplexity(text[:1000])

    # 4. Burstiness Analysis (15%)
    burstiness_score = get_burstiness_score(text)

    # 5. Fingerprint Density Check (15%)
    fingerprint_score = get_ai_fingerprint_score(text)

    # Final Weighted Calculation
    final_score = (
        (deberta_score * 0.30) + 
        (binoculars_score * 0.25) + 
        (perplexity_score * 0.15) + 
        (burstiness_score * 0.15) + 
        (fingerprint_score * 0.15)
    )
    
    reasoning = []
    if deberta_score > 50: reasoning.append("Neural patterns highly resemble AI structures.")
    if binoculars_score > 50: reasoning.append("Token distribution is highly predictable (low entropy).")
    if perplexity_score > 50: reasoning.append("Low perplexity indicates robotic word choice.")
    if burstiness_score > 50: reasoning.append("Low burstiness; sentence structures are too uniform.")
    if fingerprint_score > 50: reasoning.append("Detected density of common LLM vocabulary/fingerprints.")

    return round(final_score, 2), " ".join(reasoning)

def get_ai_score(text: str):
    cache_key = f"detect:{text}"
    cached = get_cache(cache_key)
    if cached:
        return cached['score']
    
    score, _ = get_ai_detailed_report(text)
    set_cache(cache_key, {'score': score})
    return score

TONE_PROMPTS = {
    "balanced": (
        "Rewrite the following text to make it indistinguishable from human writing. "
        "Use varying sentence structures (burstiness), natural idioms, and a conversational flow. "
        "Ensure it passes AI detection while maintaining the original meaning. "
        "Avoid over-used AI words like 'delve', 'meticulous', and 'comprehensive'. "
        "CRITICAL: Preserve all Markdown formatting exactly (tables, bold text, bullet points, headers)."
    ),
    "creative": (
        "Rewrite this text with a strong creative flair. Inject minor grammatical imperfections, "
        "use contractions, and vary sentence structures drastically. Use active voice, narrative hooks, "
        "and expressive vocabulary. Make it feel like a personal story or a blog post. "
        "CRITICAL: Preserve all Markdown formatting exactly (tables, bold text, bullet points, headers)."
    ),
    "academic": (
        "Rewrite this text in a formal academic tone but ensure it lacks the robotic predictability "
        "of AI. Use precise terminology, sophisticated syntax, and objective reasoning. "
        "Vary sentence lengths to avoid 'rhythmic' AI patterns. Ensure clarity and rigor. "
        "CRITICAL: Preserve all Markdown formatting exactly (tables, bold text, bullet points, headers)."
    ),
    "high-school": (
        "Rewrite this text as if written by a bright high school student. Use simpler vocabulary, "
        "occasional informal transitions, and a straightforward perspective. Avoid overly complex "
        "jargon. It should sound genuine, slightly unpolished, but coherent. "
        "CRITICAL: Preserve all Markdown formatting exactly (tables, bold text, bullet points, headers)."
    ),
    "journalistic": (
        "Rewrite this text in a professional journalistic style. Focus on facts, use a clear "
        "inverted pyramid structure, and short, punchy paragraphs. Use direct quotes where appropriate "
        "and maintain a neutral but engaging reporting voice. "
        "CRITICAL: Preserve all Markdown formatting exactly (tables, bold text, bullet points, headers)."
    )
}

def extract_entities(text: str):
    """Simple extraction of numbers, dates, and potential names (capitalized words)."""
    # Numbers and basic dates
    numbers = set(re.findall(r'\b\d+(?:[.,]\d+)*\b', text))
    # Capitalized words (potential names/proper nouns)
    names = set(re.findall(r'\b[A-Z][a-z]+\b', text))
    return {"numbers": numbers, "names": names}

def verify_facts(original: str, humanized: str):
    """Compares entities to find potential hallucinations or omissions."""
    orig_entities = extract_entities(original)
    hum_entities = extract_entities(humanized)
    
    issues = []
    
    missing_numbers = orig_entities["numbers"] - hum_entities["numbers"]
    added_numbers = hum_entities["numbers"] - orig_entities["numbers"]
    
    if missing_numbers:
        issues.append(f"Missing numbers: {', '.join(list(missing_numbers)[:5])}")
    if added_numbers:
        issues.append(f"New numbers detected (hallucination risk): {', '.join(list(added_numbers)[:5])}")
        
    missing_names = orig_entities["names"] - hum_entities["names"]
    if missing_names:
        # Filter out common capitalized words that might just be start of sentences
        filtered_missing = [n for n in missing_names if n.lower() not in ["the", "a", "an", "this", "that", "it"]]
        if filtered_missing:
            issues.append(f"Missing names/proper nouns: {', '.join(filtered_missing[:5])}")
            
    return issues

def chunk_text_by_tokens(text: str, max_tokens: int):
    """Splits text into chunks based on token counts."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

async def humanize_logic(text: str):
    """The 'Secret Sauce' Humanizer Loop with parallel chunk processing."""
    system_prompt = (
        "Rewrite the following text to make it indistinguishable from human writing. "
        "Use varying sentence structures (burstiness), natural idioms, and a conversational flow. "
        "Ensure it passes AI detection while maintaining the original meaning and professional quality. "
        "Avoid over-used AI words like 'delve', 'meticulous', and 'comprehensive'."
    )
    
    token_count = count_tokens(text)
    
    if token_count <= CHUNK_TOKEN_LIMIT:
        return await _process_chunk(text, system_prompt)
    
    print(f"Large text detected ({token_count} tokens). Processing chunks in parallel...")
    chunks = chunk_text_by_tokens(text, CHUNK_TOKEN_LIMIT)
    
    # Process chunks in parallel using asyncio.gather
    tasks = [_process_chunk(chunk, system_prompt) for chunk in chunks]
    humanized_chunks = await asyncio.gather(*tasks)
            
    return "\n\n".join(humanized_chunks)

async def _process_chunk(chunk_text: str, system_prompt: str):
    """Internal helper with a verification loop and caching."""
    # Cache check
    cache_key = f"humanize:{system_prompt}:{chunk_text}"
    cached = get_cache(cache_key)
    if isinstance(cached, str):
        return cached

    max_retries = 3
    target_threshold = 5.0
    current_text = chunk_text
    
    try:
        # Initial Humanization
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": current_text}
            ],
            temperature=0.8
        )
        current_text = completion.choices[0].message.content

        # Verification Loop
        for i in range(max_retries):
            score, reasoning = get_ai_detailed_report(current_text)
            
            if score < target_threshold:
                break
                
            feedback = f"The previous output was flagged with an AI probability of {score}%. "
            if reasoning:
                feedback += f"Issues identified: {reasoning} "
            feedback += "Please rewrite the text to be more human-like. Increase sentence length variation (burstiness) and use more natural, unpredictable word choices."

            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_text}, # Pass original context
                    {"role": "assistant", "content": current_text},
                    {"role": "user", "content": feedback}
                ],
                temperature=0.85 + (i * 0.05)
            )
            current_text = completion.choices[0].message.content

        set_cache(cache_key, current_text)
        return current_text
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower():
            raise HTTPException(status_code=429, detail="API Rate Limit Exceeded. Please try a smaller text or wait a minute.")
        raise e

async def stream_detection(text: str):
    try:
        cache_key = f"full_detect:{text}"
        cached = get_cache(cache_key)
        if cached:
            yield json.dumps({"log": "Retrieving results from cache...", "log_type": "success"}) + "\n"
            yield json.dumps({
                "final_result": cached['final_result']
            }) + "\n"
            return

        yield json.dumps({"log": "Initializing Ensemble Detection Engine...", "log_type": "process"}) + "\n"
        
        # Calculate sub-scores for detailed logging
        ppl = get_perplexity(text[:1000])
        yield json.dumps({"log": f"Perplexity Analysis: {round(ppl, 1)}% AI probability.", "log_type": "info"}) + "\n"
        
        bino = get_binoculars_score(text[:1000])
        yield json.dumps({"log": f"Predictability (Entropy): {round(bino, 1)}% AI probability.", "log_type": "info"}) + "\n"

        burst = get_burstiness_score(text)
        yield json.dumps({"log": f"Burstiness (Structure): {round(burst, 1)}% uniformity (AI-like).", "log_type": "info"}) + "\n"

        fing = get_ai_fingerprint_score(text)
        yield json.dumps({"log": f"AI Fingerprints: {round(fing, 1)}% common LLM vocabulary density.", "log_type": "info"}) + "\n"
        
        yield json.dumps({"log": "Running DeBERTa-v3 neural classifier...", "log_type": "process"}) + "\n"
        score = get_ai_score(text)
        
        yield json.dumps({"log": f"Final Weighted Confidence: {score}%", "log_type": "info"}) + "\n"
        
        status = "AI" if score > 50 else "Human"
        result_data = {"ai_probability": f"{score}%", "status": status}
        set_cache(cache_key, {"final_result": result_data})
        yield json.dumps({
            "final_result": result_data
        }) + "\n"
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

@app.post("/detect")
async def detect(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    return StreamingResponse(stream_detection(request.text), media_type="application/x-ndjson")

async def stream_humanization(text: str, tone: str = "balanced", allow_logging: bool = True):
    try:
        system_prompt = TONE_PROMPTS.get(tone, TONE_PROMPTS["balanced"])
        yield json.dumps({"log": f"Selected Tone: {tone.capitalize()}", "log_type": "info"}) + "\n"
        
        token_count = count_tokens(text)
        yield json.dumps({"log": f"Input analysis: {token_count} tokens detected.", "log_type": "info"}) + "\n"
        
        # Sentence-Level Surgical Humanization Strategy
        yield json.dumps({"log": "Analyzing sentence-level patterns...", "log_type": "process"}) + "\n"
        
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 4:
            yield json.dumps({"log": f"Identifying most robotic segments among {len(sentences)} sentences...", "log_type": "info"}) + "\n"
            
            sentence_scores = []
            for i, s in enumerate(sentences):
                if not s.strip() or len(s.strip()) < 15: continue 
                score = get_ai_score(s)
                sentence_scores.append({"index": i, "text": s, "score": score})
            
            # Sort by highest AI score
            sentence_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Pick top segments to rewrite
            # Limit to 3 sentences or roughly 25% of the text to preserve human parts
            num_to_rewrite = max(1, min(3, len(sentence_scores) // 2))
            targets_to_rewrite = sentence_scores[:num_to_rewrite]
            target_indices = {item["index"] for item in targets_to_rewrite}
            
            yield json.dumps({"log": f"Targeting {len(target_indices)} specific robotic segments for rewriting.", "log_type": "warning"}) + "\n"
            
            humanized_sentences = list(sentences)
            tasks = []
            target_info = []

            for item in targets_to_rewrite:
                idx = item["index"]
                orig_s = item["text"]
                yield json.dumps({"log": f"Queuing segment {idx+1} for parallel rewrite (Score: {item['score']}%)", "log_type": "process"}) + "\n"
                
                # Contextual prompt for surgical rewrite
                context_hint = " Context: " + (" ".join(sentences[max(0, idx-1):idx+2]))
                sentence_prompt = system_prompt + " Rewrite ONLY the target sentence provided below. Preserve its surrounding meaning." + context_hint[:200]
                
                tasks.append(_process_chunk(orig_s, sentence_prompt))
                target_info.append(idx)

            # Parallel Execution
            results = await asyncio.gather(*tasks)
            for i, result in enumerate(results):
                humanized_sentences[target_info[i]] = result
            
            humanized = " ".join(humanized_sentences)
        else:
            # Fallback to full block rewrite for short texts
            yield json.dumps({"log": "Text too short for surgical precision. Rewriting block...", "log_type": "process"}) + "\n"
            humanized = await _process_chunk(text, system_prompt)

        yield json.dumps({"log": "Running Fact-Preservation check...", "log_type": "process"}) + "\n"
        fact_issues = verify_facts(text, humanized)
        if fact_issues:
            for issue in fact_issues:
                yield json.dumps({"log": f"Data Integrity Warning: {issue}", "log_type": "warning"}) + "\n"
        else:
            yield json.dumps({"log": "Fact check passed: All numbers and names preserved.", "log_type": "success"}) + "\n"

        yield json.dumps({"log": "Calculating final AI score...", "log_type": "process"}) + "\n"
        new_score = get_ai_score(humanized)
        
        # Log for continuous learning if permission is granted and result is "successful" (< 20% AI)
        try:
            score_val = float(new_score.strip('%'))
            if score_val < 20:
                log_for_finetuning(text, humanized, tone, allow_logging)
        except:
            pass

        yield json.dumps({
            "final_result": {
                "original_text": text,
                "humanized_text": humanized,
                "new_ai_score": f"{new_score}%",
                "fact_issues": fact_issues
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
    return StreamingResponse(stream_humanization(request.text, request.tone, request.allow_logging), media_type="application/x-ndjson")

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

@app.post("/analyze")
async def analyze_full(request: TextRequest):
    """Provides a detailed breakdown of AI characteristics and sentence-level heatmap."""
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    cache_key = f"analyze:{text}"
    cached = get_cache(cache_key)
    if cached:
        return cached

    # Feature scores for "Why is this AI?" breakdown
    ppl = get_perplexity(text[:1000])
    bino = get_binoculars_score(text[:1000])
    burst = get_burstiness_score(text)
    fing = get_ai_fingerprint_score(text)
    final_score, reasoning = get_ai_detailed_report(text)
    
    # Sentence splitting for heatmap
    sentences = re.split(r'(?<=[.!?])\s+', text)
    heatmap = []
    for s in sentences:
        if not s.strip() or len(s.strip()) < 5: 
            if s.strip(): heatmap.append({"text": s, "score": 0})
            continue
        s_score = get_ai_score(s)
        heatmap.append({"text": s, "score": s_score})
        
    response_data = {
        "final_score": final_score,
        "metrics": {
            "Perplexity (Randomness)": f"{round(100 - ppl, 1)}%",
            "Predictability (Entropy)": f"{round(bino, 1)}%",
            "Uniformity (Burstiness)": f"{round(burst, 1)}%",
            "AI Vocabulary Density": f"{round(fing, 1)}%"
        },
        "reasoning": reasoning,
        "heatmap": heatmap
    }
    set_cache(cache_key, response_data)
    return response_data

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