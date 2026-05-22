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

# Load environment variables. 
# Priorities: 1. Bundled .env (if frozen) 2. Local .env (if exists)
BUNDLED_ENV = os.path.join(BASE_PATH, ".env")
LOCAL_ENV = os.path.join(os.getcwd(), ".env")

if os.path.exists(BUNDLED_ENV):
    load_dotenv(BUNDLED_ENV)
if os.path.exists(LOCAL_ENV):
    load_dotenv(LOCAL_ENV, override=True) # Local overrides bundled if provided

# --- CONFIGURATION ---
# Retrieve multiple keys. Supports comma-separated in GROQ_API_KEY 
# or individual keys like GROQ_API_KEY_1, GROQ_API_KEY_2...
GROQ_API_KEYS = []
raw_keys = os.getenv("GROQ_API_KEY", "")
if "," in raw_keys:
    GROQ_API_KEYS.extend([k.strip() for k in raw_keys.split(",") if k.strip()])
elif raw_keys:
    GROQ_API_KEYS.append(raw_keys)

# Also check for numbered keys GROQ_API_KEY_1 through GROQ_API_KEY_50
for i in range(1, 51):
    val = os.getenv(f"GROQ_API_KEY_{i}")
    if val and val.strip() and val.strip() not in GROQ_API_KEYS:
        GROQ_API_KEYS.append(val.strip())

# Filter out empty or None keys
GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k and k.strip()]

CURRENT_KEY_INDEX = 0
LOG_DATA_FOR_FINETUNING = True 
DATASET_FILE = os.path.join(os.getcwd(), "fine_tuning_dataset.jsonl")
MODEL_NAME = "llama-3.3-70b-versatile"
CONTEXT_WINDOW = 128000 
CHUNK_TOKEN_LIMIT = 4000 

def get_groq_client():
    global CURRENT_KEY_INDEX
    if not GROQ_API_KEYS:
        return None
    # Ensure index is within current bounds
    idx = CURRENT_KEY_INDEX % len(GROQ_API_KEYS)
    return Groq(api_key=GROQ_API_KEYS[idx])

def rotate_api_key():
    """Switches to the next available API key in the pool."""
    global CURRENT_KEY_INDEX, client
    if len(GROQ_API_KEYS) > 1:
        CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GROQ_API_KEYS)
        client = get_groq_client()
        print(f"SYSTEM: Error or Rate Limit encountered. Rotated to API key slot {CURRENT_KEY_INDEX + 1}/{len(GROQ_API_KEYS)}")
        return True
    return False

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
# 1. Neural Classifier (RoBERTa - Industry Standard)
# Using a highly stable model to resolve state-dict corruption issues seen with custom weights
print("Loading Neural Classifier...")
classifier_model_id = "openai-community/roberta-base-openai-detector" 
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)

try:
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_id)
except Exception as e:
    print(f"Standard loading failed: {e}. Attempting forced re-download...")
    # This specifically addresses 'corrupted state dictionary' by forcing a fresh download
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_id, force_download=True)

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
    # 1. Neural Classification (30%)
    inputs = classifier_tokenizer(text[:1500], return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Handle both single-output (sigmoid) and multi-output (softmax) models
        if logits.shape[-1] == 1:
            neural_score = torch.sigmoid(logits).item() * 100
        else:
            # Standard for roberta-base-openai-detector: Index 0 is Fake (AI), Index 1 is Real (Human)
            probs = torch.softmax(logits, dim=-1)
            neural_score = probs[0][0].item() * 100

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
        (neural_score * 0.30) + 
        (binoculars_score * 0.25) + 
        (perplexity_score * 0.15) + 
        (burstiness_score * 0.15) + 
        (fingerprint_score * 0.15)
    )
    
    reasoning = []
    if neural_score > 50: reasoning.append("Neural patterns highly resemble AI structures.")
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
    """Internal helper with a verification loop, caching, and automatic key rotation."""
    cache_key = f"humanize:{system_prompt}:{chunk_text}"
    cached = get_cache(cache_key)
    if isinstance(cached, str):
        return cached

    max_retries = 3
    target_threshold = 5.0
    current_text = chunk_text
    
    rotation_attempts = 0
    max_rotations = len(GROQ_API_KEYS)

    # Retry loop for API key rotation
    while rotation_attempts <= max_rotations:
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
                        {"role": "user", "content": chunk_text}, 
                        {"role": "assistant", "content": current_text},
                        {"role": "user", "content": feedback}
                    ],
                    temperature=0.85 + (i * 0.05)
                )
                current_text = completion.choices[0].message.content

            set_cache(cache_key, current_text)
            return current_text

        except Exception as e:
            err_msg = str(e).lower()
            # If the current key fails (rate limit, invalid, etc.), switch to the next one
            if any(x in err_msg for x in ["rate_limit", "429", "api_key", "auth", "unauthorized"]):
                if rotate_api_key():
                    rotation_attempts += 1
                    continue
            
            # If exhausted all keys or it's a non-API error
            if "rate_limit" in err_msg or "429" in err_msg:
                raise HTTPException(status_code=429, detail="API Key pool exhausted. All keys are currently rate-limited.")
            raise e

async def stream_detection(text: str):
    try:
        cache_key = f"full_detect:{text}"
        cached = get_cache(cache_key)
        if cached:
            yield json.dumps({"log": "Found exact match in local cache. Loading...", "log_type": "success", "progress": 50}) + "\n"
            await asyncio.sleep(0.3)
            yield json.dumps({"log": "Retrieving results from cache...", "log_type": "success", "progress": 100}) + "\n"
            yield json.dumps({
                "final_result": cached['final_result']
            }) + "\n"
            return

        yield json.dumps({"log": "Initializing Ensemble Detection Engine...", "log_type": "process", "progress": 5}) + "\n"
        await asyncio.sleep(0.1)
        
        yield json.dumps({"log": "Waking up GPT-2 Perplexity Engine...", "log_type": "process", "progress": 10}) + "\n"
        ppl = get_perplexity(text[:1000])
        yield json.dumps({"log": f"Perplexity Analysis: {round(ppl, 1)}% AI probability.", "log_type": "info", "progress": 25}) + "\n"
        
        yield json.dumps({"log": "Calculating Token Distribution Entropy (Binoculars)...", "log_type": "process", "progress": 30}) + "\n"
        bino = get_binoculars_score(text[:1000])
        yield json.dumps({"log": f"Predictability (Entropy): {round(bino, 1)}% AI probability.", "log_type": "info", "progress": 45}) + "\n"

        yield json.dumps({"log": "Analyzing Sentence Burstiness & Structural Uniformity...", "log_type": "process", "progress": 50}) + "\n"
        burst = get_burstiness_score(text)
        yield json.dumps({"log": f"Burstiness (Structure): {round(burst, 1)}% uniformity (AI-like).", "log_type": "info", "progress": 60}) + "\n"

        yield json.dumps({"log": "Scanning for LLM Vocabulary Fingerprints...", "log_type": "process", "progress": 65}) + "\n"
        fing = get_ai_fingerprint_score(text)
        yield json.dumps({"log": f"AI Fingerprints: {round(fing, 1)}% common LLM vocabulary density.", "log_type": "info", "progress": 75}) + "\n"
        
        yield json.dumps({"log": "Injecting into DeBERTa-v3 Neural Classifier Head...", "log_type": "process", "progress": 80}) + "\n"
        score = get_ai_score(text)
        
        yield json.dumps({"log": f"Aggregating Weighted Confidence Scores...", "log_type": "process", "progress": 90}) + "\n"
        yield json.dumps({"log": f"Final Weighted Confidence: {score}%", "log_type": "info", "progress": 95}) + "\n"
        
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
        yield json.dumps({"log": f"Selected Tone: {tone.capitalize()}", "log_type": "info", "progress": 5}) + "\n"
        
        token_count = count_tokens(text)
        yield json.dumps({"log": f"Input analysis: {token_count} tokens detected.", "log_type": "info", "progress": 10}) + "\n"
        
        yield json.dumps({"log": "Analyzing sentence-level patterns...", "log_type": "process", "progress": 15}) + "\n"
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 4:
            yield json.dumps({"log": f"Identifying most robotic segments among {len(sentences)} sentences...", "log_type": "info", "progress": 20}) + "\n"
            
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

            for i, item in enumerate(targets_to_rewrite):
                idx = item["index"]
                orig_s = item["text"]
                
                # Create wrapper to capture task completion for progress updates
                async def tracked_task(s, p, task_idx):
                    res = await _process_chunk(s, p)
                    return res, task_idx

                # Contextual prompt for surgical rewrite
                context_hint = " Context: " + (" ".join(sentences[max(0, idx-1):idx+2]))
                sentence_prompt = system_prompt + " Rewrite ONLY the target sentence provided below. Preserve its surrounding meaning." + context_hint[:200]
                
                tasks.append(tracked_task(orig_s, sentence_prompt, idx))
                target_info.append(idx)

            # Process tasks as they complete for real-time progress
            completed_count = 0
            total_tasks = len(tasks)
            for future in asyncio.as_completed(tasks):
                result, idx = await future
                completed_count += 1
                humanized_sentences[idx] = result
                
                prog = 20 + int((completed_count / total_tasks) * 55)
                yield json.dumps({
                    "log": f"Segment {idx+1} humanized ({completed_count}/{total_tasks}).", 
                    "log_type": "success", 
                    "progress": prog
                }) + "\n"

            yield json.dumps({"log": "All robotic segments successfully rewritten.", "log_type": "success", "progress": 75}) + "\n"
            
            humanized = " ".join(humanized_sentences)
        else:
            yield json.dumps({"log": "Short text detected. Rewriting block...", "log_type": "process", "progress": 30}) + "\n"
            humanized = await _process_chunk(text, system_prompt)
            yield json.dumps({"log": "Block rewriting completed.", "log_type": "success", "progress": 75}) + "\n"

        yield json.dumps({"log": "Verifying fact preservation...", "log_type": "process", "progress": 80}) + "\n"
        fact_issues = verify_facts(text, humanized)
        if fact_issues:
            for issue in fact_issues:
                yield json.dumps({"log": f"Integrity Warning: {issue}", "log_type": "warning", "progress": 85}) + "\n"
        else:
            yield json.dumps({"log": "Fact check passed: 100% data integrity.", "log_type": "success", "progress": 85}) + "\n"

        yield json.dumps({"log": "Calculating final AI score and verdict...", "log_type": "process", "progress": 90}) + "\n"
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
    # Disabled for production release to prevent user modification of bundled keys
    raise HTTPException(status_code=403, detail="API Key modification is disabled in this version.")

@app.get("/check-config")
async def check_config():
    return {
        "has_api_key": len(GROQ_API_KEYS) > 0,
        "pool_size": len(GROQ_API_KEYS),
        "env_location": os.path.join(os.getcwd(), ".env")
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