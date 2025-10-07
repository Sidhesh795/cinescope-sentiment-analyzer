from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
from src.preprocess import clean_text
import csv
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Load model and vectorizer with error handling
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Models loaded successfully!")
except FileNotFoundError:
    print("⚠️  Model files not found. Please run train_model.py first.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    vectorizer = None

REVIEWS_FILE = 'movie_reviews_aggregate.json'
USER_REVIEWS_CSV = 'user_reviews.csv'

def interpret_sentiment(prob):
    if prob > 0.85:
        return {'label': 'Highly Positive', 'emoji': '🌟', 
                'description': 'Exceptional! Loved the movie.', 'color': '#10b981'}
    elif prob > 0.65:
        return {'label': 'Positive', 'emoji': '😊', 
                'description': 'Enjoyed it!', 'color': '#3b82f6'}
    elif prob > 0.45:
        return {'label': 'Mixed', 'emoji': '😐', 
                'description': 'Neutral feelings.', 'color': '#f59e0b'}
    elif prob > 0.25:
        return {'label': 'Negative', 'emoji': '😞', 
                'description': 'Disappointed.', 'color': '#ef4444'}
    else:
        return {'label': 'Highly Negative', 'emoji': '😡', 
                'description': 'Strongly disliked.', 'color': '#dc2626'}

def load_movie_stats():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_movie_stats(stats):
    with open(REVIEWS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

def update_movie_stats(movie_name, sentiment_prob):
    stats = load_movie_stats()
    
    if movie_name not in stats:
        stats[movie_name] = {
            'positive': 0, 'negative': 0, 'total': 0, 'avg_score': 0, 'reviews': []
        }
    
    if sentiment_prob >= 0.5:
        stats[movie_name]['positive'] += 1
    else:
        stats[movie_name]['negative'] += 1
    
    stats[movie_name]['total'] += 1
    stats[movie_name]['reviews'].append({
        'score': round(sentiment_prob, 3),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    total_score = sum([r['score'] for r in stats[movie_name]['reviews']])
    stats[movie_name]['avg_score'] = round(total_score / stats[movie_name]['total'], 3)
    
    save_movie_stats(stats)
    return stats[movie_name]

@app.route('/')
def home():
    all_stats = load_movie_stats()
    return render_template('index.html', all_movies=all_stats)

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form.get('review', '').strip()
    movie_name = request.form.get('movie_name', '').strip()
    
    if not review_text or not movie_name:
        flash("Please provide both movie name and review", "error")
        return redirect(url_for('home'))
    
    # Check if models are loaded
    if model is None or vectorizer is None:
        flash("Models not loaded. Please train the model first by running train_model.py", "error")
        return redirect(url_for('home'))
    
    try:
        cleaned_review = clean_text(review_text)
        vec = vectorizer.transform([cleaned_review])
        prob = model.predict_proba(vec)[0][1]
        
        sentiment_result = interpret_sentiment(prob)
        movie_stats = update_movie_stats(movie_name, prob)
        
        # Log review
        file_exists = os.path.exists(USER_REVIEWS_CSV)
        with open(USER_REVIEWS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Movie', 'Review', 'Probability', 'Sentiment'])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                movie_name, review_text, round(prob, 3), sentiment_result['label']
        ])
    
        all_stats = load_movie_stats()
        
        return render_template('index.html',
                             movie_name=movie_name,
                             review=review_text,
                             sentiment=sentiment_result,
                             confidence=round(prob * 100, 1),
                             movie_stats=movie_stats,
                             all_movies=all_stats,
                             show_result=True)
    
    except Exception as e:
        flash(f"Error processing review: {str(e)}", "error")
        return redirect(url_for('home'))

if __name__ == '__main__':
    # Initialize files
    if not os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'w') as f:
            json.dump({}, f)
    
    print("\n" + "="*60)
    print("🎬 CineScope Movie Sentiment Analyzer")
    print("="*60)
    print("\n🚀 Starting server...")
    print("📍 Open your browser and go to: http://127.0.0.1:5000")
    print("\n⏹️  Press CTRL+C to stop the server\n")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)