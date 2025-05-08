from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from utils import clean_tweet, extract_twitter_features, explain_prediction, predict_with_rules
import os

app = Flask(__name__)

# Load models with enhanced model fallback
def load_models():
    try:
        # Try to load enhanced models first
        with open('model/enhanced_tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        with open('model/enhanced_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('model/enhanced_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
            
        print("Enhanced models loaded successfully!")
        return tfidf_vectorizer, scaler, model, True  # True indicates enhanced model
        
    except FileNotFoundError:
        # Fall back to original models
        with open('model/tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('model/classifier.pkl', 'rb') as f:
            model = pickle.load(f)
            
        print("Standard models loaded successfully")
        return tfidf_vectorizer, scaler, model, False  # False indicates standard model

# Check if models exist
if os.path.exists('model/classifier.pkl') or os.path.exists('model/enhanced_classifier.pkl'):
    tfidf_vectorizer, scaler, model, is_enhanced = load_models()
    print(f"Using {'enhanced' if is_enhanced else 'standard'} classification model")
else:
    print("Models not found. Please run train_model.py first")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get input tweet
    tweet = request.form.get('tweet', '')
    
    if not tweet:
        return jsonify({
            'error': 'No tweet provided'
        })
    
    # Use rule-based enhanced prediction
    prediction, prob_dict = predict_with_rules(tweet, model, tfidf_vectorizer, scaler)
    
    # Clean tweet for explanation purposes
    cleaned_tweet = clean_tweet(tweet)
    text_features = tfidf_vectorizer.transform([cleaned_tweet])
    
    # Get top contributing words
    explanation = explain_prediction(text_features, tweet, model, tfidf_vectorizer)
    
    # Check if we're analyzing appearance-related comments
    appearance_terms = ['weight', 'look', 'looks', 'fat', 'thin', 'beautiful', 'pretty']
    has_appearance_terms = any(term in tweet.lower().split() for term in appearance_terms)
    
    # Add note for appearance-related comments
    if has_appearance_terms and 'note' not in explanation:
        explanation['note'] = 'Comments about appearance can be difficult to classify without full context.'
    
    # Return results
    return jsonify({
        'tweet': tweet,
        'prediction': prediction,
        'probabilities': prob_dict,
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(debug=True)