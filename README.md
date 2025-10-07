# 🎬 CineScope - AI Movie Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

Ultra-modern movie review sentiment analysis powered by Machine Learning. Features a sleek dark UI inspired by Rotten Tomatoes and Letterboxd.

![CineScope Demo](https://via.placeholder.com/800x400?text=Add+Your+Screenshot+Here)

## ✨ Features

- 🤖 **89% Accuracy** - Trained on 50,000 IMDB reviews
- 📊 **Aggregate Statistics** - Track positive/negative reviews per movie
- 🎨 **Modern Dark UI** - Cyberpunk-inspired design with neon accents
- ⚡ **Real-time Analysis** - Instant sentiment detection
- 🔄 **Incremental Learning** - Model improves over time with user input
- 📈 **Visual Analytics** - Beautiful charts and progress bars

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/Sidhesh795/cinescope-sentiment-analyzer.git
cd cinescope-sentiment-analyzer

# 2. Navigate to project folder
cd movie_sentiment_app

# 3. Create virtual environment
python -m venv venv

# 4. Activate virtual environment
# Windows:
venv\Scripts\activate.bat
# Mac/Linux:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
