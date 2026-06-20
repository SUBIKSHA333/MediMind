import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from models.symptom_classifier import classify_symptoms

# Sample labeled test dataset (symptom text -> true condition)
test_data = [
    ("I have severe headache and blurred vision", "Migraine"),
    ("My head is pounding and lights bother my eyes", "Migraine"),
    ("I feel very thirsty and need to urinate frequently", "Diabetes"),
    ("My blood sugar feels high and I'm always tired", "Diabetes"),
    ("I have a runny nose and sore throat", "Common Cold"),
    ("I'm sneezing a lot with mild cough", "Common Cold"),
    ("My chest feels tight and I'm wheezing", "Asthma"),
    ("I have shortness of breath and coughing", "Asthma"),
    ("My blood pressure feels high with headaches", "Hypertension"),
    ("I have chest pain and dizziness", "Hypertension"),
    ("I feel nauseous after eating and have stomach cramps", "Food Poisoning"),
    ("I vomited after my last meal and feel sick", "Food Poisoning"),
    ("I feel constantly worried and my heart races", "Anxiety"),
    ("I can't stop overthinking and feel restless", "Anxiety"),
]

def evaluate_with_stratified_kfold(n_splits=3):
    """Evaluate classifier performance using Stratified K-Fold cross-validation"""
    
    texts = [item[0] for item in test_data]
    labels = [item[1] for item in test_data]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accuracies = []
    all_true = []
    all_pred = []
    
    print(f"🔍 Running Stratified {n_splits}-Fold Cross-Validation on Symptom Classifier\n")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
        fold_true = [labels[i] for i in test_idx]
        fold_pred = []
        
        for i in test_idx:
            predicted_label, score = classify_symptoms(texts[i])
            fold_pred.append(predicted_label)
        
        fold_acc = accuracy_score(fold_true, fold_pred)
        fold_accuracies.append(fold_acc)
        
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)
        
        print(f"Fold {fold}: Accuracy = {fold_acc*100:.1f}%")
    
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\n📊 Average Accuracy across folds: {avg_accuracy*100:.1f}% (± {std_accuracy*100:.1f}%)")
    print(f"\n📋 Full Classification Report:\n")
    print(classification_report(all_true, all_pred, zero_division=0))
    
    return avg_accuracy, std_accuracy

if __name__ == "__main__":
    evaluate_with_stratified_kfold(n_splits=2)