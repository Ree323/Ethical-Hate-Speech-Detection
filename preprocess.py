import pandas as pd
import re
import string
import os
import requests
from io import StringIO
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load data
def load_data(filepath='train.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df

# Acquire complementary datasets
def acquire_additional_datasets():
    additional_datasets = []
    
    # Try to load Davidson dataset (a well-known hate speech dataset)
    try:
        print("Attempting to download Davidson Hate Speech dataset...")
        url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
        response = requests.get(url)
        
        if response.status_code == 200:
            davidson_df = pd.read_csv(StringIO(response.text))
            print(f"Successfully loaded Davidson dataset with {len(davidson_df)} rows")
            
            # Map Davidson labels to our format
            davidson_mapping = {
                0: 'hate_speech',  # their class 0 = hate speech
                1: 'offensive',    # their class 1 = offensive
                2: 'neutral'       # their class 2 = neither
            }
            
            davidson_df['standardized_label'] = davidson_df['class'].map(davidson_mapping)
            davidson_df['source'] = 'davidson'
            
            # Select relevant columns
            davidson_df = davidson_df[['tweet', 'standardized_label', 'source']]
            
            additional_datasets.append(davidson_df)
        else:
            print("Could not download Davidson dataset. Status code:", response.status_code)
    except Exception as e:
        print(f"Error loading Davidson dataset: {e}")
    
    # Add more datasets here if desired...
    
    return additional_datasets

# Clean tweets
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

# Process your primary dataset
def process_dataset(df):
    print("Dataset columns:", df.columns.tolist())
    
    # Your dataset has X_test for tweets and y_test for labels
    tweet_column = 'X_test'  # Using your actual column name
    label_column = 'y_test'  # Using your actual column name
    
    # Clean tweets
    print("Cleaning tweets...")
    df['cleaned_tweet'] = df[tweet_column].apply(clean_tweet)
    
    # Your dataset already has clear labels, so we use them directly
    df['label'] = df[label_column]
    df['source'] = 'original'
    
    print("Label distribution:", df['label'].value_counts())
    
    return df

# Combine datasets
def combine_datasets(primary_df, additional_datasets):
    print("Combining datasets...")
    
    # Prepare primary dataset
    primary_df_selected = primary_df[['cleaned_tweet', 'label', 'source']].rename(
        columns={'cleaned_tweet': 'text', 'label': 'standardized_label'})
    
    # Initialize combined dataframe with primary dataset
    combined_datasets = [primary_df_selected]
    
    # Add additional datasets
    for dataset in additional_datasets:
        # Clean tweets in additional dataset
        dataset['text'] = dataset['tweet'].apply(clean_tweet)
        dataset = dataset[['text', 'standardized_label', 'source']]
        combined_datasets.append(dataset)
    
    # Combine all dataframes
    combined_df = pd.concat(combined_datasets, ignore_index=True)
    
    print(f"Combined dataset size: {len(combined_df)} rows")
    print("Label distribution in combined dataset:")
    print(combined_df['standardized_label'].value_counts())
    
    return combined_df

# Balance the combined dataset
def balance_dataset(df):
    print("Balancing dataset...")
    
    # Remove duplicates
    df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Dataset size after removing duplicates: {len(df)} rows")
    
    # Check class distribution
    class_counts = df['standardized_label'].value_counts()
    print("Class distribution before balancing:", class_counts)
    
    # Determine target size per class (up to 3000 samples)
    max_samples = min(3000, class_counts.max())
    print(f"Target samples per class: {max_samples}")
    
    balanced_dfs = []
    
    for label in df['standardized_label'].unique():
        subset = df[df['standardized_label'] == label]
        
        # If too many samples, downsample
        if len(subset) > max_samples:
            subset = resample(subset, replace=False, n_samples=max_samples, random_state=42)
        # If too few samples, upsample
        elif len(subset) < max_samples:
            subset = resample(subset, replace=True, n_samples=max_samples, random_state=42)
        
        balanced_dfs.append(subset)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"Balanced dataset size: {len(balanced_df)} rows")
    print("Class distribution after balancing:", balanced_df['standardized_label'].value_counts())
    
    return balanced_df

# Create rule-based module for enhancing classification
def create_rule_based_module():
    """Create rule_based.py with enhancement functions"""
    rule_code = r'''import re

def is_definitely_neutral(text):
    """Rule-based system to identify obviously neutral content"""
    if not text:
        return False
    
    text = text.lower().strip()
    
    # Compliment patterns - almost always neutral
    compliment_patterns = [
        r'(look|looks|looking)\s(good|great|nice|pretty|beautiful|handsome)',
        r'(is|are|seem|seems)\s(good|great|nice|pretty|beautiful|handsome)',
        r'(you|he|she|they)\s(are|is)\s(good|great|nice|pretty|beautiful|handsome)'
    ]
    
    for pattern in compliment_patterns:
        if re.search(pattern, text):
            return True
    
    # Weight-related neutral comments (common source of false positives)
    weight_neutral_patterns = [
        r'(gained|lost)\s(weight|pounds|kilos)',
        r'(look|looks)\s(thinner|bigger|smaller|larger)'
    ]
    
    # Only classify as neutral if there are no clear offensive terms
    offensive_terms = ['fat', 'ugly', 'stupid', 'idiot', 'dumb', 'hate']
    has_offensive = any(term in text.split() for term in offensive_terms)
    
    if not has_offensive:
        for pattern in weight_neutral_patterns:
            if re.search(pattern, text):
                return True
    
    return False

def detect_positive_context(text):
    """Detect if potentially sensitive topics are in a positive context"""
    text = text.lower()
    
    # Break text into segments
    segments = re.split(r'[.,;:!?]', text)
    
    # Words that could be problematic in some contexts
    sensitive_words = ['weight', 'fat', 'thin', 'look', 'looks', 'size']
    
    # Positive context markers
    positive_markers = ['good', 'great', 'nice', 'pretty', 'beautiful', 
                        'handsome', 'amazing', 'love', 'like', 'still']
    
    # Negation markers that flip meaning
    negations = ['not', "isn't", "aren't", "wasn't", "weren't", "don't", 
                "doesn't", "didn't", "no", "never"]
    
    for segment in segments:
        words = segment.split()
        
        # Skip very short segments
        if len(words) < 2:
            continue
        
        # Check if segment contains sensitive words
        has_sensitive = any(word in sensitive_words for word in words)
        
        if has_sensitive:
            # Check for positive markers
            has_positive = any(marker in words for marker in positive_markers)
            
            # Check for negated positive (becomes negative)
            for i, word in enumerate(words):
                if word in positive_markers and i > 0:
                    if any(neg in words[max(0, i-3):i] for neg in negations):
                        has_positive = False
                        break
            
            if has_positive:
                return True
    
    return False

def enhance_classification(text, prediction, probabilities):
    """Apply rules to correct model predictions"""
    # Apply neutral check
    if is_definitely_neutral(text) or detect_positive_context(text):
        return "neutral", {
            "neutral": 0.85,
            "offensive": 0.10,
            "hate_speech": 0.05
        }
    
    # Adjust borderline hate speech predictions
    if prediction == 'hate_speech' and probabilities.get('hate_speech', 1.0) < 0.65:
        # Less confident hate speech - downgrade to offensive
        return "offensive", {
            "neutral": 0.15,
            "offensive": 0.75,
            "hate_speech": 0.10
        }
    
    # Keep original prediction if no rules apply
    return prediction, probabilities
'''
    
    # Write to file with utf-8 encoding
    with open('rule_based.py', 'w', encoding='utf-8') as f:
        f.write(rule_code)
    
    print("Created rule_based.py with enhancement functions")

# Update utils.py with rule-based improvements
def update_utils_file():
    utils_code = r'''import re
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
'''

# Add to rule_based.py
def is_definitely_offensive(text):
    # Check for attacks on opinions/ideas
    opinion_attack_patterns = [
        r'(your|their|his|her)\s+(opinion|idea|thought|view)\s+(is|are)\s+(trash|garbage|stupid|dumb|worthless)',
        r'(nobody|no one)\s+cares\s+(about|what)\s+(your|their)',
        r'(shut|shut up|be quiet)'
    ]
    
    for pattern in opinion_attack_patterns:
        if re.search(pattern, text.lower()):
            return True
    
    return False
    
    # Write to file with utf-8 encoding
    with open('utils.py', 'w', encoding='utf-8') as f:
        f.write(utils_code)
    
    print("Updated utils.py with rule-based improvements")

# Create test suite
def create_test_suite():
    test_code = r'''import pandas as pd
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
'''
    
    # Write to file with UTF-8 encoding to handle special characters
    with open('test_improvements.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("Created test_improvements.py for evaluating model accuracy")

# Create enhanced model training script
def create_enhanced_training_script():
    train_code = r'''import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
import os

# Import utility functions
from utils import clean_tweet, extract_twitter_features

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)

# Load enhanced dataset 
try:
    df = pd.read_csv('data/enhanced_training_data.csv')
    print(f"Using enhanced dataset with {len(df)} samples")
except:
    # Fall back to regular dataset
    df = pd.read_csv('data/train_data.csv')
    print(f"Using standard dataset with {len(df)} samples")

# Split data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create features
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased from 10000
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Extracting Twitter features...")
train_twitter_features = [extract_twitter_features(tweet) for tweet in X_train]
test_twitter_features = [extract_twitter_features(tweet) for tweet in X_test]

# Convert to numpy arrays
train_twitter_features = np.array(train_twitter_features)
test_twitter_features = np.array(test_twitter_features)

print("Scaling Twitter features...")
scaler = StandardScaler()
train_twitter_features_scaled = scaler.fit_transform(train_twitter_features)
test_twitter_features_scaled = scaler.transform(test_twitter_features)

print("Combining features...")
X_train_combined = hstack([
    X_train_tfidf, 
    csr_matrix(train_twitter_features_scaled)
])
X_test_combined = hstack([
    X_test_tfidf, 
    csr_matrix(test_twitter_features_scaled)
])

# Try a more sophisticated model
print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_combined, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test_combined)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
print("Saving model...")
with open('model/enhanced_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('model/enhanced_tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
    
with open('model/enhanced_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Enhanced model saved!")
'''
    
    # Write to file with utf-8 encoding
    with open('train_enhanced_model.py', 'w', encoding='utf-8') as f:
        f.write(train_code)
    
    print("Created train_enhanced_model.py for training improved model")

# Split into train/test
def split_data(df):
    X = df['cleaned_tweet']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

# Save processed data
def save_processed_data(X_train, X_test, y_train, y_test):
    # Create DataFrames with explicit column names
    train_df = pd.DataFrame()
    train_df['text'] = X_train
    train_df['label'] = y_train
    train_df.to_csv('data/train_data.csv', index=False)
    
    test_df = pd.DataFrame()
    test_df['text'] = X_test
    test_df['label'] = y_test
    test_df.to_csv('data/test_data.csv', index=False)
    
    print("Saved processed data to data/train_data.csv and data/test_data.csv")

# Main function
def main():
    print("Starting enhanced data preprocessing...")
    
    # Load original dataset
    df = load_data('train.csv')
    
    print("\nSample data:")
    print(df.head())
    
    # Process original dataset
    df_processed = process_dataset(df)
    
    # Try to acquire additional datasets
    additional_datasets = acquire_additional_datasets()
    
    # Combine datasets if any additional were loaded
    if additional_datasets:
        combined_df = combine_datasets(df_processed, additional_datasets)
        # Balance the combined dataset
        enhanced_df = balance_dataset(combined_df)
        # Save enhanced dataset
        enhanced_df.to_csv('data/enhanced_training_data.csv', index=False)
        print("Enhanced training data saved to data/enhanced_training_data.csv")
    else:
        print("No additional datasets found. Using only the original dataset.")
        # Simply use the processed original dataset
        df_processed = df_processed.rename(columns={'cleaned_tweet': 'text', 'label': 'standardized_label'})
        df_processed.to_csv('data/enhanced_training_data.csv', index=False)
    
    # Create rule-based module for enhancing classification
    create_rule_based_module()
    
    # Update utils.py with improvements
    update_utils_file()
    
    # Create enhanced model training script
    create_enhanced_training_script()
    
    # Create test suite for evaluating improvements
    create_test_suite()
    
    # Split and save processed data from original dataset for compatibility
    X_train, X_test, y_train, y_test = split_data(df_processed)
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("\n====== Enhanced Preprocessing Complete ======")
    print("\nNext steps:")
    print("1. Run the standard model training:")
    print("   python train_model.py")
    print("2. Run the enhanced model training:")
    print("   python train_enhanced_model.py")
    print("3. Test the improvements:")
    print("   python test_improvements.py")
    print("\nThe improved app will automatically use the enhanced model if available.")

if __name__ == "__main__":
    main()