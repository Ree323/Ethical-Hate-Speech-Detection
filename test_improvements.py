import pandas as pd
from utils import clean_tweet, extract_twitter_features
from rule_based import enhance_classification
import pickle
from scipy.sparse import hstack, csr_matrix
import numpy as np

# Load models
try:
    with open('model/enhanced_tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open('model/enhanced_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('model/enhanced_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Using enhanced model for testing")
except FileNotFoundError:
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open('model/classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("Using original model for testing")

# Define test cases that were previously misclassified
test_cases = [
    {"text": "she looks beautiful today", "expected": "neutral"},
    {"text": "he gained weight but still looks good", "expected": "neutral"},
    {"text": "your hair looks different", "expected": "neutral"},
    {"text": "that outfit makes you look bigger", "expected": "neutral"},
    {"text": "you're acting stupid today", "expected": "offensive"},
    {"text": "I hate people who look like you", "expected": "hate_speech"}
]

# Run tests
success_count = 0
total_tests = len(test_cases)

print("\nRunning test cases...\n")
print("-" * 60)

for i, case in enumerate(test_cases):
    text = case["text"]
    expected = case["expected"]
    
    # Get raw model prediction first
    cleaned_text = clean_tweet(text)
    text_features = tfidf_vectorizer.transform([cleaned_text])
    twitter_features = extract_twitter_features(text)
    twitter_features_scaled = scaler.transform([twitter_features])
    combined_features = hstack([text_features, csr_matrix(twitter_features_scaled)])
    
    raw_prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    prob_dict = {model.classes_[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    # Get enhanced prediction with rules
    enhanced_prediction, enhanced_probs = enhance_classification(text, raw_prediction, prob_dict)
    
    # Check if test passed
    is_success = enhanced_prediction == expected
    if is_success:
        success_count += 1
    
    # Print results
    print(f"Test Case {i+1}:")
    print(f"Text: {text}")
    print(f"Expected: {expected}")
    print(f"Raw model prediction: {raw_prediction}")
    print(f"Rule-enhanced prediction: {enhanced_prediction}")
    print(f"Success: {'Pass' if is_success else 'Fail'}")
    print("-" * 60)

print(f"\nTest Results: {success_count}/{total_tests} passed ({success_count/total_tests*100:.1f}%)")
