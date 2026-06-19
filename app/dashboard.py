import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import ask_llm
from nlp.nlp_pipeline import analyze_query
from models.symptom_classifier import classify_symptoms
from database.db_setup import get_session, PatientSession, SymptomPrediction
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="MediMind", page_icon="🩺", layout="wide")

st.title("🩺 MediMind — AI Medical Intelligence Platform")
st.caption("RAG + LLM + NLP + Hugging Face powered medical assistant")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["💬 Chat Assistant", "🔍 Symptom Checker", "📊 Analytics"])

# ---------------- CHAT PAGE ----------------
if page == "💬 Chat Assistant":
    st.header("Ask MediMind a Question")
    
    user_input = st.text_input("Type your medical question:")
    
    if st.button("Ask"):
        if user_input:
            with st.spinner("Analyzing and generating response..."):
                nlp_result = analyze_query(user_input)
                response = ask_llm(user_input)
                
                # Save to database
                session = get_session()
                new_session = PatientSession(
                    user_query=user_input,
                    ai_response=response,
                    intent="general_query",
                    sentiment=nlp_result["sentiment"]
                )
                session.add(new_session)
                session.commit()
                session.close()
            
            st.success("Response generated!")
            st.write(response)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment Detected", nlp_result["sentiment"])
            with col2:
                st.metric("Confidence", f"{nlp_result['confidence']*100:.1f}%")

# ---------------- SYMPTOM CHECKER PAGE ----------------
elif page == "🔍 Symptom Checker":
    st.header("AI Symptom Checker")
    
    symptoms = st.text_area("Describe your symptoms:")
    
    if st.button("Check Symptoms"):
        if symptoms:
            with st.spinner("Analyzing symptoms..."):
                label, score = classify_symptoms(symptoms)
                
                session = get_session()
                new_prediction = SymptomPrediction(
                    symptoms=symptoms,
                    predicted_disease=label,
                    confidence=f"{score*100:.1f}%"
                )
                session.add(new_prediction)
                session.commit()
                session.close()
            
            st.success(f"Predicted Condition: **{label}**")
            st.progress(score)
            st.write(f"Confidence: {score*100:.1f}%")

# ---------------- ANALYTICS PAGE ----------------
elif page == "📊 Analytics":
    st.header("Session Analytics")
    
    session = get_session()
    predictions = session.query(SymptomPrediction).all()
    chats = session.query(PatientSession).all()
    session.close()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Symptom Checks", len(predictions))
    col2.metric("Total Chat Sessions", len(chats))
    
    if predictions:
        df = pd.DataFrame([{"condition": p.predicted_disease} for p in predictions])
        fig = px.histogram(df, x="condition", title="Most Common Predicted Conditions")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No symptom check data yet. Try the Symptom Checker page!")