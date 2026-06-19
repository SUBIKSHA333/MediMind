from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import ask_llm
from nlp.nlp_pipeline import analyze_query
from models.symptom_classifier import classify_symptoms
from database.db_setup import get_session, PatientSession, SymptomPrediction

app = FastAPI(title="MediMind API")

# Request format for chat
class ChatRequest(BaseModel):
    message: str

# Request format for symptom check
class SymptomRequest(BaseModel):
    symptoms: str

@app.get("/")
def home():
    return {"message": "MediMind API is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    """Main chat endpoint — analyzes query and responds using LLM"""
    
    # Run NLP analysis
    nlp_result = analyze_query(request.message)
    
    # Get AI response
    ai_response = ask_llm(request.message)
    
    # Save to database
    session = get_session()
    new_session = PatientSession(
        user_query=request.message,
        ai_response=ai_response,
        intent="general_query",
        sentiment=nlp_result["sentiment"]
    )
    session.add(new_session)
    session.commit()
    session.close()
    
    return {
        "response": ai_response,
        "sentiment": nlp_result["sentiment"],
        "entities": nlp_result["entities"]
    }

@app.post("/check-symptoms")
def check_symptoms(request: SymptomRequest):
    """Symptom classification endpoint"""
    
    label, score = classify_symptoms(request.symptoms)
    
    # Save to database
    session = get_session()
    new_prediction = SymptomPrediction(
        symptoms=request.symptoms,
        predicted_disease=label,
        confidence=f"{score*100:.1f}%"
    )
    session.add(new_prediction)
    session.commit()
    session.close()
    
    return {
        "predicted_condition": label,
        "confidence": f"{score*100:.1f}%"
    }