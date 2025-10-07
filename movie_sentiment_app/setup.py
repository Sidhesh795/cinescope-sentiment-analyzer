#!/usr/bin/env python3
"""
Setup script for Movie Sentiment Analysis Application
This script helps set up the environment and train the initial model.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        return False
    logger.info(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install requirements: {e}")
        return False

def check_dataset():
    """Check if the IMDB dataset exists"""
    data_path = os.path.join("data", "IMDB Dataset.csv")
    if os.path.exists(data_path):
        logger.info("✅ IMDB Dataset found")
        return True
    else:
        logger.warning("⚠️  IMDB Dataset not found")
        logger.info("Please download the IMDB Dataset from:")
        logger.info("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        logger.info(f"Place it as: {data_path}")
        return False

def train_model():
    """Train the sentiment analysis model"""
    logger.info("Training the sentiment analysis model...")
    try:
        # Change to src directory to run training
        os.chdir("src")
        subprocess.check_call([sys.executable, "train_model.py"])
        os.chdir("..")  # Go back to root
        logger.info("✅ Model training completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Model training failed: {e}")
        os.chdir("..")  # Ensure we're back in root even if training fails
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during training: {e}")
        os.chdir("..")
        return False

def main():
    """Main setup function"""
    logger.info("="*60)
    logger.info("🎬 Movie Sentiment Analysis - Setup")
    logger.info("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        logger.error("Setup failed at requirements installation")
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        logger.warning("Setup completed with warnings. Please download the dataset to continue.")
        sys.exit(1)
    
    # Train model
    if not train_model():
        logger.error("Setup failed at model training")
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("🎉 Setup completed successfully!")
    logger.info("="*60)
    logger.info("You can now run the application with:")
    logger.info("python app.py")
    logger.info("="*60)

if __name__ == "__main__":
    main()
