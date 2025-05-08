import re
import string
import numpy as np
from scipy.sparse import hstack, csr_matrix
from rule_based import enhance_classification

# Clean tweet function for prediction
def clean_tweet(tweet):
    # Convert to lowercase
    tweet = str(tweet).lower()
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove hashtag symbol but keep text
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    # Remove RT indicator
    tweet = re.sub(r'^rt\s+', '', tweet)
    # Remove punctuation
    tweet = ''.join([c for c in tweet if c not in string.punctuation])
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

# Extract Twitter features
def extract_twitter_features(tweet):
    # Convert to string if not already
    tweet = str(tweet)
    
    # Extract features
    has_mention = 1 if '@' in tweet else 0
    has_hashtag = 1 if '#' in tweet else 0
    has_url = 1 if 'http' in tweet else 0
    is_retweet = 1 if tweet.lower().startswith('rt @') else 0
    exclamation_count = tweet.count('!')
    question_count = tweet.count('?')
    upper_case_ratio = sum(1 for c in tweet if c.isupper()) / max(len(tweet), 1)
    
    return [
        has_mention, has_hashtag, has_url, is_retweet,
        exclamation_count, question_count, upper_case_ratio
    ]

# Enhanced prediction with rules
def predict_with_rules(tweet, model, vectorizer, scaler):
    """Apply ML model with rule-based corrections"""
    # Standard prediction process
    cleaned_tweet = clean_tweet(tweet)
    text_features = vectorizer.transform([cleaned_tweet])
    twitter_features = extract_twitter_features(tweet)
    twitter_features_scaled = scaler.transform([twitter_features])
    combined_features = hstack([text_features, csr_matrix(twitter_features_scaled)])
    
    # Get model prediction
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    prob_dict = {model.classes_[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    # Apply rule-based corrections
    enhanced_prediction, enhanced_probs = enhance_classification(tweet, prediction, prob_dict)
    
    return enhanced_prediction, enhanced_probs

# Explanation function
def explain_prediction(features, text, model, vectorizer):
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # For multi-class models, get coefficients for the predicted class
    if hasattr(model, 'coef_'):
        if len(model.classes_) <= 2:
            coefficients = model.coef_[0]
        else:
            # For multinomial, get coefficients for each class
            coefficients = {}
            for i, label in enumerate(model.classes_):
                coefficients[label] = model.coef_[i, :len(feature_names)]
    else:
        return {"error": "Model doesn't provide feature coefficients"}
    
    # Get words in the text
    words = set(text.lower().split())
    
    # Find matching words that appear in the vectorizer's vocabulary
    word_indices = {word: idx for idx, word in enumerate(feature_names) if word in words}
    
    # Get top contributing words
    contributing_words = []
    
    if isinstance(coefficients, dict):
        # For multiclass
        for label, coef in coefficients.items():
            class_words = []
            for word, idx in word_indices.items():
                class_words.append((word, float(coef[idx])))
            
            # Sort by importance
            class_words.sort(key=lambda x: abs(x[1]), reverse=True)
            contributing_words.append({
                'class': label,
                'words': class_words[:5]
            })
    else:
        # For binary
        word_weights = [(word, float(coefficients[idx])) for word, idx in word_indices.items()]
        word_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        contributing_words = word_weights[:5]
    
    # Add note for appearance-related comments
    if any(word in text.lower() for word in ['weight', 'look', 'fat', 'thin']):
        return {
            'contributing_words': contributing_words,
            'note': 'Comments about appearance can be difficult to classify without full context.'
        }
    
    return {'contributing_words': contributing_words}
