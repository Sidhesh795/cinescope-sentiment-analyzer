import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys
from preprocess import clean_text
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*50)
    logger.info("Starting Model Training")
    logger.info("="*50)
    
    # Load dataset
    try:
        logger.info("Loading dataset...")
        # Get the absolute path to the data directory
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Handle case when __file__ is not defined (like when running with exec)
            current_dir = os.getcwd()
            
        data_path = os.path.join(current_dir, '..', 'data', 'IMDB Dataset.csv')
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        data = None
        
        for encoding in encodings:
            try:
                data = pd.read_csv(data_path, encoding=encoding)
                logger.info(f"Dataset loaded successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            raise ValueError("Could not read dataset with any supported encoding")
            
        logger.info(f"Dataset loaded: {len(data)} reviews")
    except FileNotFoundError:
        logger.error("Dataset not found! Please download 'IMDB Dataset.csv'")
        logger.error("Place it in: data/IMDB Dataset.csv")
        sys.exit(1)
    
    # Preprocess
    logger.info("Preprocessing reviews...")
    data['review'] = data['review'].apply(clean_text)
    data = data[data['review'].str.len() > 0]
    
    # Map sentiments
    sentiment_map = {'positive': 1, 'negative': 0}
    data['sentiment'] = data['sentiment'].map(sentiment_map)
    data = data.dropna(subset=['sentiment'])
    
    logger.info(f"Preprocessing complete. {len(data)} reviews remaining.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['review'], data['sentiment'], 
        test_size=0.2, random_state=42, stratify=data['sentiment']
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create features
    logger.info("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    
    logger.info(f"\nTest Accuracy: {accuracy:.4f}")
    logger.info(f"\n{classification_report(y_test, predictions, target_names=['Negative', 'Positive'])}")
    
    # Save model
    logger.info("Saving model and vectorizer...")
    # Save models in the project root directory
    try:
        project_root = os.path.join(current_dir, '..')
    except NameError:
        project_root = '..'
    
    model_path = os.path.join(project_root, 'model.pkl')
    vectorizer_path = os.path.join(project_root, 'vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    logger.info("✅ Training complete! Model saved.")

if __name__ == "__main__":
    main()