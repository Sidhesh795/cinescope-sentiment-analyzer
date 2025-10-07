# 🎬 CineScope - AI Movie Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

Ultra-modern movie review sentiment analysis powered by Machine Learning. Features a sleek dark UI inspired by Rotten Tomatoes and Letterboxd.

## ✨ Features

- 🤖 **89% Accuracy** - Trained on 50,000 IMDB reviews
- 📊 **Aggregate Statistics** - Track positive/negative reviews per movie
- 🎨 **Modern Dark UI** - Cyberpunk-inspired design with neon accents
- ⚡ **Real-time Analysis** - Instant sentiment detection
- 🔄 **Incremental Learning** - Model improves over time with user input
- 📈 **Visual Analytics** - Beautiful charts and progress bars

## 📋 Table of Contents

- Prerequisites
- Installation Steps
- Dataset Download
- Project Structure
- Training the Model
- Running the Application
- Usage Flow
- Tech Stack
-  Deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation
```bash
# Step 1: Clone the Repository
bash
# git clone https://github.com/Sidhesh795/cinescope-sentiment-analyzer.git
cd cinescope-sentiment-analyzer
cd movie_sentiment_app

# Step 2: Create Virtual Environment
bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate.bat

# For Mac/Linux:
source venv/bin/activate

# Your prompt should now show (venv)

# Step 3: Install Dependencies
bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

```
## 📊 Dataset Download (IMPORTANT!)

⚠️ CRITICAL STEP: You must download the dataset before training the model!
Option 1: Manual Download (Recommended)
- Visit Kaggle Dataset Page:

  (IMDB-Dataset-of-50k-Movie-Reviews)[https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews]]

- Login to Kaggle (create free account if needed)
- Click "Download" button (downloads a ZIP file ~30MB)
- Extract the ZIP file - you'll get IMDB_Dataset.csv (81MB)
- Create data folder in project root:
```
bash
mkdir data
```
- Move the CSV file to the data folder:
```
bash
   # Windows
   move IMDB_Dataset.csv data\

   # Mac/Linux
   mv IMDB_Dataset.csv data/
```
## Option 2: Using Kaggle API
```
bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (get API token from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Unzip
unzip imdb-dataset-of-50k-movie-reviews.zip -d data/

## Verify Dataset
bash# Check if file exists
# Windows:
dir data\IMDB_Dataset.csv

# Mac/Linux:
ls -lh data/IMDB_Dataset.csv

# Should show: IMDB_Dataset.csv (approximately 81 MB)
```
## Dataset Details:

- 📁 Filename: IMDB_Dataset.csv
- 📏 Size: 81 MB
- 📊 Rows: 50,000 reviews
- 📋 Columns: review, sentiment
- ⚖️ Balance: 25,000 positive + 25,000 negative

### 📁 Project Structure
After completing all setup steps, your folder structure should look like this:
cinescope-sentiment-analyzer/
│
├── README.md                           # This file
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore rules
│
└── movie_sentiment_app/                # Main application folder
    │
    ├── data/                           # Dataset folder
    │   └── IMDB_Dataset.csv           # ⬇️ DOWNLOAD THIS FIRST!
    │
    ├── src/                            # Source code
    │   ├── __init__.py
    │   ├── preprocess.py              # Text preprocessing functions
    │   ├── train_model.py             # Model training script
    │   └── predict.py                 # CLI prediction tool (optional)
    │
    ├── templates/                      # HTML templates
    │   └── index.html                 # Main web interface
    │
    ├── static/                         # Static files (auto-created)
    │   ├── css/
    │   └── js/
    │
    ├── venv/                           # Virtual environment (not in repo)
    │   ├── Scripts/                   # Windows
    │   ├── bin/                       # Mac/Linux
    │   └── Lib/
    │
    ├── app.py                          # Flask application (main entry point)
    ├── requirements.txt                # Python dependencies
    ├── setup.py                        # Automated setup script
    ├── render.yaml                     # Deployment configuration
    │
    ├── model.pkl                       # 🤖 Trained model (created after training)
    ├── vectorizer.pkl                  # 🤖 TF-IDF vectorizer (created after training)
    │ 
    ├── movie_reviews_aggregate.json    # 📊 Movie statistics (auto-created)
    ├── user_reviews.csv                # 📊 User review logs (auto-created)
    ├── learning_data.csv               # 📊 Learning data (auto-created)
    └── retrain_log.json                # 📊 Retrain history (auto-created)

## Files Explanation
--------------------------------------------------------------------------------------
|       File/Folder      |            Purpose             |      Created Bydata      |
|------------------------------------------------------------------------------------|
| IMDB_Dataset.csv       |   Training dataset             |   YOU (download)         |
| src/preprocess.py      |   Text cleaning                |   Provided               |
| src/train_model.py     |   Model training               |   Provided               |
| app.py                 |   Flask web server             |   Provided               |
| templates/index.html   |   Web interface                |   Provided               |
| model.pkl              |   Trained model                |   Training script        |
| vectorizer.pkl         |   TF-IDF vectorizer            |   Training script        |
| venv/                  |   Isolated Python environment  |   You (python -m venv)   |
| *.json, *.csv          |   Runtime data                 |   Application            |
--------------------------------------------------------------------------------------

## 🤖 Training the Model
⚠️ Make sure you've downloaded the dataset first!
```
bash# Navigate to src folder
cd src

# Run training script (takes 1-3 minutes)
python train_model.py

# Expected output:
# ============================================================
# Starting Model Training
# ============================================================
# Loading dataset...
# ✓ Dataset loaded: 50000 reviews
# ...
# ✓ Test Accuracy: 0.8945
# ✅ SUCCESS! Training complete!
# Files created: model.pkl, vectorizer.pkl

# Go back to main folder
cd ..
```
## What happens during training:

- ✅ Loads 50,000 IMDB reviews
- ✅ Preprocesses text (cleaning, tokenization)
- ✅ Splits into train/test (80/20)
- ✅ Creates TF-IDF features
- ✅ Trains Logistic Regression model
- ✅ Evaluates accuracy (~89%)
- ✅ Saves model.pkl and vectorizer.pkl


##🎬 Running the Application
```
bash# Make sure you're in movie_sentiment_app folder
# and virtual environment is activated (venv)

# Run the Flask app
python app.py

# Expected output:
# ============================================================
# 🎬 CineScope Movie Sentiment Analyzer
# ============================================================
# 🚀 Starting server...
# 📍 Open: http://127.0.0.1:5000
# ⏹️  Press CTRL+C to stop
# ============================================================
```
# Open your browser and navigate to:
(http://127.0.0.1:5000)
or
(http://localhost:5000)

## 🔄 Usage Flow
For First-Time Users:
1. Clone Repository
   ↓
2. Create Virtual Environment
   ↓
3. Install Dependencies
   ↓
4. Download Dataset from Kaggle ⚠️ CRITICAL
   ↓
5. Place Dataset in data/ folder
   ↓
6. Train Model (cd src && python train_model.py)
   ↓
7. Run Application (python app.py)
   ↓
8. Open Browser (http://127.0.0.1:5000)
   ↓
9. Enter Movie Name + Review
   ↓
10. Get Instant Sentiment Analysis! 🎉

## Application Workflow:
User Input (Movie Name + Review)
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression Model
        ↓
Sentiment Prediction (0-100%)
        ↓
Interpretation (Highly Positive → Highly Negative)
        ↓
Update Aggregate Statistics
        ↓
Display Results + Charts
        ↓
Log to CSV + JSON

## 🛠️ Tech Stack
Backend
- Flask 3.0 - Lightweight web framework
- Python 3.12 - Programming language
- Scikit-learn 1.3+ - Machine learning library
- NLTK 3.8+ - Natural language processing
- Pandas 2.0+ - Data manipulation
- NumPy 1.24+ - Numerical computing
- Joblib 1.3+ - Model serialization

Frontend
- HTML5 - Structure
- CSS3 - Styling (Dark theme, gradients, animations)
- Vanilla JavaScript - Dynamic interactions

Data Storage
- JSON - Aggregate movie statistics
- CSV - Individual review logs

Machine Learning
- Algorithm: Logistic Regression
- Vectorization: TF-IDF (5000 features, bigrams)
- Accuracy: ~89%
- Training Time: 1-3 minutes

## 📊 Model Performance
Metric                Value 
Accuracy              89.45%
Precision (Positive)  0.90
Recall (Positive)     0.89
F1-Score (Positive)   0.89
Precision (Negative)  0.89
Recall (Negative)     0.90
F1-Score (Negative)   0.89

## 🎯 Sentiment Categories
CategoryProbability RangeDescription🌟 Highly Positive85-100%Exceptional praise, loved the movie😊 Positive65-85%Enjoyed it, strong liking😐 Mixed45-65%Neutral feelings, pros and cons😞 Negative25-45%Disappointed, several issues😡 Highly Negative0-25%Strongly disliked, very critical

## 🚀 Deployment
# Deploy to Render
- Push code to GitHub
- Create account on Render.com
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Render auto-detects render.yaml
- Click "Create Web Service"
- Wait 5-10 minutes for deployment

# Deploy to Railway
```
bash
npm install -g @railway/cli
railway login
railway init
railway up
```
# Deploy to Heroku
```
bash
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create cinescope-app
git push heroku main
```
## 🔧 Configuration
# Customize Sentiment Thresholds
Edit app.py:
```
python
def interpret_sentiment(prob):
    if prob > 0.90:  # Change from 0.85
        return {...}
```
# Tune Model Parameters
Edit src/train_model.py:
```
python
# Increase features
vectorizer = TfidfVectorizer(max_features=10000)

# Adjust regularization
model = LogisticRegression(C=0.5, max_iter=1000)
```
## 📈 Future Enhancements
- BERT/Transformer models
- User authentication system
- Export analysis as PDF
- Movie poster integration (TMDb API)
- Sentiment trends over time
- Multi-language support
- Mobile app version
- Batch review processing
- REST API for external integrations


## 🐛 Troubleshooting
- Issue: ModuleNotFoundError
- Solution: Activate virtual environment and install dependencies
```
bash
venv\Scripts\activate.bat
pip install -r requirements.txt
```
- Issue: Dataset not found
- Solution: Download IMDB_Dataset.csv and place in data/ folder

- Issue: Model not found
- Solution: Train the model first
```
bash
cd src
python train_model.py
cd ..
```
- Issue: Port already in use
- Solution: Change port in app.py:
```
python
app.run(debug=True, port=5001)  # Use different port
```
## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open Pull Request


## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author
Sidhesh
- GitHub: @Sidhesh795
- Project: CineScope Sentiment Analyzer


## 🙏 Acknowledgments

- IMDB Dataset from Kaggle
- Scikit-learn team
- Flask framework
- Inspired by Rotten Tomatoes and Letterboxd

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!

## 📧 Contact
For questions or feedback, please open an issue on GitHub.

***Built with ❤️ and Python***
