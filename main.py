import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from groq import Groq
import os

# --- CONFIGURATION ---
# Get a free API key from https://console.groq.com/
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" 
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="AI Detector & Humanizer API")

# Load the AI Detector Model (RoBERTa-base-openai-detector)
print("Loading AI Detector model... please wait.")
detector_pipe = pipeline("text-classification", model="roberta-base-openai-detector")

class TextRequest(BaseModel):
    text: str

def get_ai_score(text: str):
    """Returns the probability of the text being AI-generated."""
    # Truncate text to 512 tokens for the model
    results = detector_pipe(text[:512])
    # The model returns labels like 'Real' or 'Fake'
    # We want to return the 'Fake' (AI) score
    score = 0
    for res in results:
        if res['label'] == 'Fake':
            score = res['score']
        else:
            score = 1 - res['score']
    return round(score * 100, 2)

def humanize_logic(text: str):
    """The 'Secret Sauce' Humanizer Loop"""
    system_prompt = (
        "Rewrite the following text to make it indistinguishable from human writing. "
        "Use varying sentence structures (burstiness), natural idioms, and a conversational flow. "
        "Ensure it passes AI detection while maintaining the original meaning and professional quality. "
        "Avoid over-used AI words like 'delve', 'meticulous', and 'comprehensive'."
    )
    
    # 1st Pass Humanization
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0.8, # Higher temp = more human-like randomness
    )
    
    humanized_text = completion.choices[0].message.content
    
    # Verification Loop: Check if it still looks like AI
    score = get_ai_score(humanized_text)
    
    # If score is still high (>30%), do one more aggressive pass
    if score > 30:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "The previous attempt still looks like AI. Rewrite this again. Be more creative, change sentence lengths more drastically, and use less predictable vocabulary."},
                {"role": "user", "content": humanized_text}
            ],
            temperature=0.9,
        )
        humanized_text = completion.choices[0].message.content

    return humanized_text

@app.post("/detect")
async def detect(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    score = get_ai_score(request.text)
    return {"ai_probability": f"{score}%", "status": "AI" if score > 50 else "Human"}

@app.post("/humanize")
async def humanize(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    humanized = humanize_logic(request.text)
    new_score = get_ai_score(humanized)
    
    return {
        "original_text": request.text,
        "humanized_text": humanized,
        "new_ai_score": f"{new_score}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)