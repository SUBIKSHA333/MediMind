from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection
DB_URL = "mysql+pymysql://root:Subikshagd_3@localhost/medimind"

# Create engine
engine = create_engine(DB_URL)

# Base class
Base = declarative_base()

# Table 1 — Patient Sessions
class PatientSession(Base):
    __tablename__ = "patient_sessions"
    
    id = Column(Integer, primary_key=True)
    user_query = Column(Text)
    ai_response = Column(Text)
    intent = Column(String(100))
    sentiment = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)

# Table 2 — Symptom Predictions
class SymptomPrediction(Base):
    __tablename__ = "symptom_predictions"
    
    id = Column(Integer, primary_key=True)
    symptoms = Column(Text)
    predicted_disease = Column(String(200))
    confidence = Column(String(50))
    created_at = Column(DateTime, default=datetime.now)

# Create all tables
def init_db():
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully!")

# Get database session
def get_session():
    Session = sessionmaker(bind=engine)
    return Session()

# Test the connection
if __name__ == "__main__":
    init_db()