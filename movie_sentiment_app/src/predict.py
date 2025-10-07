import joblib
import os
import sys
import numpy as np
from preprocess import clean_text
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentPredictor:
    """Movie review sentiment predictor using trained ML model"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize the predictor with model and vectorizer paths"""
        if model_path is None or vectorizer_path is None:
            # Default paths relative to project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..')
            model_path = os.path.join(project_root, 'model.pkl')
            vectorizer_path = os.path.join(project_root, 'vectorizer.pkl')
        
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            logger.info("Loading trained model and vectorizer...")
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info("Models loaded successfully!")
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            logger.error("Please run train_model.py first to create the models.")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single review text
        
        Args:
            text (str): The review text to analyze
            
        Returns:
            dict: Contains prediction, probability, and interpretation
        """
        if not text or not isinstance(text, str):
            return {
                'prediction': 'neutral',
                'probability': 0.5,
                'confidence': 'low',
                'error': 'Invalid input text'
            }
        
        try:
            # Clean the text
            cleaned_text = clean_text(text)
            
            if not cleaned_text:
                return {
                    'prediction': 'neutral',
                    'probability': 0.5,
                    'confidence': 'low',
                    'error': 'Text too short after cleaning'
                }
            
            # Vectorize the text
            text_vector = self.vectorizer.transform([cleaned_text])
            
            # Get prediction and probability
            prediction = self.model.predict(text_vector)[0]
            probability = self.model.predict_proba(text_vector)[0]
            
            # Get the probability for positive sentiment (class 1)
            positive_prob = probability[1]
            
            # Determine sentiment label
            sentiment = 'positive' if prediction == 1 else 'negative'
            
            # Determine confidence level
            confidence = self._get_confidence_level(positive_prob)
            
            return {
                'prediction': sentiment,
                'probability': float(positive_prob),
                'confidence': confidence,
                'raw_text': text,
                'cleaned_text': cleaned_text
            }
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return {
                'prediction': 'error',
                'probability': 0.0,
                'confidence': 'low',
                'error': str(e)
            }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of review texts
            
        Returns:
            list: List of prediction dictionaries
        """
        return [self.predict_sentiment(text) for text in texts]
    
    def _get_confidence_level(self, probability):
        """Determine confidence level based on probability"""
        if probability > 0.8 or probability < 0.2:
            return 'high'
        elif probability > 0.65 or probability < 0.35:
            return 'medium'
        else:
            return 'low'

def main():
    """Test the predictor with sample reviews"""
    predictor = SentimentPredictor()
    
    # Test samples
    test_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "Terrible movie. Waste of time. Poor acting and boring storyline.",
        "It was okay, nothing special but not bad either.",
        "Best movie ever! Loved every minute of it. Highly recommended!",
        "Worst film I've ever seen. Complete garbage."
    ]
    
    print("\n" + "="*60)
    print("MOVIE SENTIMENT ANALYSIS - TEST PREDICTIONS")
    print("="*60)
    
    for i, review in enumerate(test_reviews, 1):
        result = predictor.predict_sentiment(review)
        
        print(f"\nReview {i}:")
        print(f"Text: {review}")
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Confidence: {result['confidence'].upper()}")
        print("-" * 40)

if __name__ == "__main__":
    main()
