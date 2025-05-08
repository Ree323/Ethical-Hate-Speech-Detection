import pandas as pd
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
