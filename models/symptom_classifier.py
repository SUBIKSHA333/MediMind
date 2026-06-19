from transformers import pipeline

# Load a zero-shot classification model from Hugging Face
# This lets us classify symptoms into categories WITHOUT training our own model first
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Disease/condition categories we want to detect
CATEGORIES = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "Common Cold",
    "Migraine",
    "Food Poisoning",
    "Anxiety"
]

def classify_symptoms(symptom_text):
    """Takes patient symptom text and predicts the most likely condition"""
    
    result = classifier(symptom_text, CATEGORIES)
    
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    
    print(f"\n🩺 Symptom: {symptom_text}")
    print(f"🎯 Most likely condition: {top_label} ({top_score*100:.1f}% confidence)")
    
    print("\nAll predictions:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score*100:.1f}%")
    
    return top_label, top_score

# Test it
if __name__ == "__main__":
    classify_symptoms("I have been feeling very thirsty, tired, and need to urinate frequently")
    classify_symptoms("I have a runny nose, sore throat and mild cough")