import spacy
from transformers import pipeline

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def extract_entities(text):
    """Extract important entities like dates, body parts, conditions from text"""
    doc = nlp(text)
    
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})
    
    print(f"\n📝 Text: {text}")
    print(f"🏷️  Entities found:")
    for ent in entities:
        print(f"   {ent['text']} → {ent['label']}")
    
    return entities

def analyze_sentiment(text):
    """Detect if the patient sounds anxious, calm, distressed etc."""
    result = sentiment_analyzer(text)[0]
    
    label = result["label"]
    score = result["score"]
    
    print(f"😊 Sentiment: {label} ({score*100:.1f}% confidence)")
    
    return label, score

def analyze_query(text):
    """Run full NLP analysis on a patient query"""
    entities = extract_entities(text)
    sentiment, confidence = analyze_sentiment(text)
    
    return {
        "entities": entities,
        "sentiment": sentiment,
        "confidence": confidence
    }

# Test it
if __name__ == "__main__":
    analyze_query("I have had a severe headache since yesterday and I'm really worried about it")
    analyze_query("My fever is gone now and I feel much better today")