import re
import nltk
from nltk.corpus import stopwords
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    """Clean and preprocess text for sentiment analysis"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords but keep sentiment words
        stop_words = set(stopwords.words('english'))
        sentiment_words = {'not', 'no', 'nor', 'neither', 'never', 
                          'but', 'however', 'very', 'really', 'too', 'so'}
        stop_words = stop_words - sentiment_words
        
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        cleaned = ' '.join(tokens)
        return cleaned if cleaned else text.lower()
    
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text.lower() if text else ""