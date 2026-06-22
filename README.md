
# 💬 Sentiment Analysis on Twitter Data

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-4A9B6F?style=flat)](https://nltk.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TextBlob](https://img.shields.io/badge/TextBlob-3776AB?style=flat)](https://textblob.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 📌 Objective

Build a sentiment analysis tool that classifies tweets as **Positive**, **Negative**, or **Neutral** using NLP techniques. Compares a lexicon-based approach (TextBlob) with ML classifiers (Naive Bayes, SVM).

---

## 🛠️ Approaches Used

| Approach | Type | Accuracy |
|---|---|---|
| TextBlob | Lexicon-based | 85.0% |
| Naive Bayes + TF-IDF | ML Classifier | 100.0% |
| SVM + TF-IDF | ML Classifier | 100.0% |

---

## 🔧 NLP Pipeline

```
Raw Tweet
    ↓
Text Cleaning (URLs, mentions, punctuation removed)
    ↓
Stopword Removal + Lemmatization (NLTK)
    ↓
TF-IDF Vectorization (bigrams, 5000 features)
    ↓
Classification (Naive Bayes / SVM)
    ↓
Sentiment Label: Positive / Negative / Neutral
```

---

## 📸 Screenshots

### Sentiment Distribution
<img width="1692" height="741" alt="sentiment_distribution" src="https://github.com/user-attachments/assets/be702589-3cf0-4d66-8e7d-c45755557eac" />

### TextBlob vs Actual Labels
<img width="1184" height="731" alt="textblob_comparison" src="https://github.com/user-attachments/assets/2fbbe68a-48fc-4586-997c-5551d2e168f8" />


### Model Accuracy Comparison
<img width="1184" height="733" alt="model_comparison" src="https://github.com/user-attachments/assets/0c889570-387f-4dd1-a637-bbeacb56bf36" />


### Confusion Matrix
<img width="976" height="731" alt="confusion_matrix" src="https://github.com/user-attachments/assets/28616a2e-61c6-492d-ad8d-279ac4fddf67" />

---

## 📁 Project Structure

```
sentiment-analysis/
├── sentiment_analysis.py   ← Main script
├── data/
│   └── tweets.csv          ← Generated dataset (3,000 tweets)
├── screenshots/            ← All chart outputs
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/Rosesharma13/sentiment-analysis-twitter.git
cd sentiment-analysis-twitter
pip install -r requirements.txt
python sentiment_analysis.py
```

---

## 🔮 Live Prediction Demo

```python
# Example outputs
"This product is absolutely amazing! Love it!"
→ TextBlob: positive | SVM: positive

"Worst service ever, completely disappointed"
→ TextBlob: negative | SVM: negative

"The package arrived today as scheduled"
→ TextBlob: neutral | SVM: neutral
```

---

## 📈 Key Insights

- **TextBlob** works well for clearly positive/negative text (85% accuracy)
- **SVM + TF-IDF** with bigrams achieves highest accuracy on structured data
- **Neutral tweets** are the hardest to classify correctly
- **Text cleaning** (lemmatization + stopword removal) significantly improves ML model performance

---

## 🔑 Key Learnings

- NLP preprocessing pipeline — cleaning, tokenization, lemmatization
- TF-IDF vectorization with unigrams and bigrams
- Comparing lexicon-based vs ML-based sentiment approaches
- Building sklearn pipelines for text classification

---

## 👩‍💻 Author

**Rose Sharma** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rose-sharma13)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Rosesharma13)
