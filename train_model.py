import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import os

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)

# Load processed data
def load_processed_data():
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    # Check column names for debugging
    print("Train data columns:", train_df.columns.tolist())
    print("Test data columns:", test_df.columns.tolist())
    
    # Use 'text' and 'label' column names (instead of X_train, y_train)
    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']
    
    return X_train, X_test, y_train, y_test


# Extract Twitter features
def extract_twitter_features(tweets):
    features = []
    for tweet in tweets:
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
        
        features.append([
            has_mention, has_hashtag, has_url, is_retweet,
            exclamation_count, question_count, upper_case_ratio
        ])
    return features

# Feature engineering
def create_features(X_train, X_test):
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.75,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print("Extracting Twitter features...")
    train_twitter_features = extract_twitter_features(X_train)
    test_twitter_features = extract_twitter_features(X_test)
    
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
    
    return X_train_combined, X_test_combined, tfidf_vectorizer, scaler

# Train model
def train_model(X_train, y_train):
    print("Training model...")
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return y_pred

# Save model
def save_model(model, tfidf_vectorizer, scaler):
    print("Saving model...")
    with open('model/classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('model/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model saved to model/ directory")

# Main function
def main():
    print("Starting model training...")
    X_train, X_test, y_train, y_test = load_processed_data()
    X_train_combined, X_test_combined, tfidf_vectorizer, scaler = create_features(X_train, X_test)
    model = train_model(X_train_combined, y_train)
    evaluate_model(model, X_test_combined, y_test)
    save_model(model, tfidf_vectorizer, scaler)
    print("Training complete!")

if __name__ == "__main__":
    main()