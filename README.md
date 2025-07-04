# Sentiment Analysis NLP Project

## Overview
This project implements a modular and extensible Natural Language Processing (NLP) pipeline for sentiment classification of text data (e.g., tweets and social media posts) into three categories: **Positive**, **Neutral**, and **Negative**. It leverages traditional machine learning models and is structured to allow future integration of deep learning approaches.

---

## Key Features
- **Sentiment Classification**: Classifies text into Positive, Neutral, or Negative sentiments.
- **Models**: Logistic Regression, Support Vector Machine (SVM), Ridge Classifier.
- **Tech Stack**: Python, Scikit-learn, Pandas, NLTK, FastAPI (optional for deployment).
- **Evaluation Metrics**: Accuracy, F1-Score, ROC-AUC.
- **Testing**: Unit tests using `pytest` for text preprocessing and model pipeline.
- **Modular Design**: Easy to extend with advanced models like BiLSTM or BERT.

---


## Project Structure

Sentiment-Analysis-NLP/
├── data/ # Raw and processed datasets
├── models/ # Trained model files
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code
│ ├── app/ # FastAPI application
│ ├── data_preprocessing/ # Text preprocessing scripts
│ ├── models/ # Model training and evaluation scripts
│ ├── utils/ # Utility functions and helpers
├── tests/ # Unit tests for preprocessing and pipeline
├── requirements.txt # Python dependencies
├── main.py # Main script to run the training pipeline
├── .gitignore # Git ignore file
└── README.md # Project documentation


---

## Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

---

## Installation

### 1. Clone the Repository

<!-- ```bash -->
git clone https://github.com/Prabhuteja124/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP 

### 2. Create and Activate a Virtual Environment

#### On Windows
python -m venv venv
venv\Scripts\activate

#### On Linux / macOS
python3 -m venv venv
source venv/bin/activate

### 3. Install Dependencies
- pip install -r requirements.txt

## Usage

### 1. Update Data Paths
 - Ensure that all file paths in the src/ modules (especially for loading data and saving models) are correct to avoid errors.

### 2. Run the Training Pipeline
 - python main.py
 - Alternatively, use pre-trained models stored in the models/ directory for inference or experimentation.

### 3. Run Unit Tests
 - pytest tests/



## Future Work
- Integrate deep learning models (e.g., BiLSTM, BERT) for improved accuracy.
- Enhance text preprocessing to handle emojis, abbreviations, and informal language.
- Build an API using FastAPI to serve predictions as a web service.
- Create a user interface using Streamlit or another dashboard framework for live interaction.

### Author
##Prabhu Teja

📧 Email: prabhuteja124@gmail.com
🔗 GitHub: github.com/Prabhuteja124
 