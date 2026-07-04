"""
Sentiment Analysis on Twitter Data
Codec Technologies AI Internship Project
Author: Rose Sharma

Classifies tweets as Positive, Negative, or Neutral using NLP.
Approaches: TextBlob (lexicon-based) + Naive Bayes + SVM
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from xquik_export import load_xquik_export

# Download required NLTK data
for pkg in ['stopwords', 'punkt', 'wordnet']:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

np.random.seed(42)
os.makedirs('screenshots', exist_ok=True)
os.makedirs('data', exist_ok=True)

PALETTE = ["#1a3c6e", "#e74c3c", "#27ae60"]

# ─── Generate realistic tweet dataset ────────────────────────────
POSITIVE_TWEETS = [
    "Just had the best coffee ever! Loving this morning ☕😊",
    "Amazing product! Highly recommend to everyone!",
    "Great customer service, very helpful and friendly staff",
    "This is absolutely wonderful, exceeded all my expectations!",
    "So happy with my purchase, fast delivery too!",
    "Love this app, makes my life so much easier!",
    "Fantastic experience at the restaurant, will definitely return",
    "Best day ever! Everything went perfectly today 🎉",
    "The new update is incredible, so many great features",
    "Grateful for amazing friends and family today 💕",
    "Just finished an amazing book, totally recommend it!",
    "Perfect weather today, feeling blessed and happy",
    "Thrilled with the results, hard work pays off!",
    "Outstanding quality, worth every penny spent",
    "Excellent service as always, never disappoints",
    "So excited for the weekend! Plans are amazing",
    "This made my day so much better, thank you!",
    "Brilliant solution to a complex problem, well done!",
    "Impressed with how fast the support team responded",
    "Life is good when you have great people around you",
]

NEGATIVE_TWEETS = [
    "Terrible customer service, waited 2 hours for help",
    "This product broke after one day, complete waste of money",
    "Very disappointed with the quality, not as described",
    "Worst experience ever, will never come back again",
    "Delivery was late and the package was damaged",
    "Horrible app, keeps crashing every 5 minutes",
    "Awful food, cold and tasteless, total disappointment",
    "Fed up with these constant delays and poor service",
    "This is absolutely unacceptable, I want a refund",
    "Disgusting behavior from staff, very rude and unhelpful",
    "Complete failure, nothing works as advertised",
    "So frustrated with this company, no response to emails",
    "Terrible quality for such a high price, not worth it",
    "The worst movie I have ever watched, total waste of time",
    "Extremely disappointed, this did not meet basic expectations",
    "Broken on arrival, no return policy, total scam",
    "Customer support is useless, nobody answers",
    "This ruined my entire day, absolutely unacceptable",
    "Poor design, difficult to use, and crashes constantly",
    "Never again! This company does not care about customers",
]

NEUTRAL_TWEETS = [
    "Just woke up and checking the news",
    "Heading to the office today for some meetings",
    "The weather forecast says it might rain tomorrow",
    "Watched a documentary about climate change last night",
    "Reading about the latest technology developments",
    "The new policy will take effect from next month",
    "Traffic was normal today on the highway",
    "Attended a conference on machine learning today",
    "The quarterly report will be released next week",
    "Updating my software to the latest version",
    "Had a regular lunch at the usual place today",
    "The meeting was rescheduled to next Tuesday",
    "Looking at options for my next phone upgrade",
    "The store closes at 9pm on weekdays",
    "Comparing different plans before making a decision",
    "The new model comes with updated specifications",
    "Checking the schedule for the upcoming events",
    "The office will be closed on public holidays",
    "Reviewing the terms and conditions of the service",
    "The package is expected to arrive in 3-5 business days",
]

def generate_dataset(n=3000):
    rows = []
    templates = {
        'positive': POSITIVE_TWEETS,
        'negative': NEGATIVE_TWEETS,
        'neutral':  NEUTRAL_TWEETS
    }

    per_class = n // 3
    for label, tweets in templates.items():
        for i in range(per_class):
            base = tweets[i % len(tweets)]
            # Add slight variation
            variations = [
                base,
                base + " #trending",
                base.lower(),
                "RT: " + base,
                base + " 🙌" if label == 'positive' else base,
            ]
            rows.append({
                'tweet': np.random.choice(variations),
                'sentiment': label
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─── Text cleaning ────────────────────────────────────────────────
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'@\w+', '', text)                  # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)             # Keep hashtag words
    text = re.sub(r'[^a-z\s]', '', text)              # Keep letters only
    text = re.sub(r'\s+', ' ', text).strip()          # Clean whitespace
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
        return ' '.join(tokens)
    except:
        return text


# ─── TextBlob sentiment ───────────────────────────────────────────
def get_textblob_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:   return 'positive'
    elif polarity < -0.1: return 'negative'
    else:                 return 'neutral'


# ─── Plotting ─────────────────────────────────────────────────────
def plot_sentiment_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Sentiment Analysis — Overview', fontsize=14, fontweight='bold')

    counts = df['sentiment'].value_counts()
    axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=PALETTE, startangle=90, textprops={'fontsize': 11})
    axes[0].set_title('Sentiment Distribution', fontweight='bold')

    axes[1].bar(counts.index, counts.values, color=PALETTE)
    axes[1].set_title('Tweet Count by Sentiment', fontweight='bold')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('screenshots/sentiment_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Distribution chart saved")


def plot_model_comparison(results):
    names = list(results.keys())
    accs  = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=PALETTE[:len(names)], width=0.5)
    ax.set_ylim(0.5, 1.0)
    ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.spines[['top','right']].set_visible(False)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('screenshots/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Model comparison saved")


def plot_confusion_matrix(cm, model_name, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('screenshots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved")


def plot_textblob_vs_actual(df):
    df['tb_sentiment'] = df['tweet'].apply(get_textblob_sentiment)
    match = (df['tb_sentiment'] == df['sentiment']).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    comparison = pd.crosstab(df['sentiment'], df['tb_sentiment'])
    comparison.plot(kind='bar', ax=ax, color=PALETTE, rot=0)
    ax.set_title(f'TextBlob vs Actual Labels (Match: {match:.1%})', fontweight='bold')
    ax.set_xlabel('Actual Sentiment')
    ax.set_ylabel('Count')
    ax.legend(title='TextBlob Prediction')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig('screenshots/textblob_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ TextBlob comparison saved")
    return match


# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("SENTIMENT ANALYSIS — Codec Technologies")
    print("=" * 55)

    # Generate data
    print("\n📊 Generating tweet dataset...")
    df = generate_dataset(3000)
    df.to_csv('data/tweets.csv', index=False)
    print(f"✅ Dataset: {len(df):,} tweets")
    print(df['sentiment'].value_counts().to_string())

    # EDA
    plot_sentiment_distribution(df)

    # TextBlob analysis
    print("\n🔍 Running TextBlob analysis...")
    tb_match = plot_textblob_vs_actual(df)
    print(f"   TextBlob accuracy: {tb_match:.1%}")

    # ML models
    print("\n🤖 Training ML models...")
    df['clean_tweet'] = df['tweet'].apply(clean_tweet)

    X = df['clean_tweet']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    labels = ['negative', 'neutral', 'positive']
    ml_results = {}

    # Naive Bayes
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    nb_pred = nb_pipeline.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    ml_results['Naive Bayes'] = nb_acc
    print(f"  Naive Bayes accuracy: {nb_acc:.3f}")

    # SVM
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ('clf', LinearSVC(random_state=42, max_iter=2000))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    ml_results['SVM'] = svm_acc
    print(f"  SVM accuracy:        {svm_acc:.3f}")

    xquik_export_path = os.getenv("XQUIK_EXPORT_PATH", "").strip()
    if xquik_export_path:
        xquik_df = pd.DataFrame(load_xquik_export(xquik_export_path))
        xquik_df["clean_tweet"] = xquik_df["tweet"].apply(clean_tweet)
        xquik_df["textblob_sentiment"] = xquik_df["tweet"].apply(get_textblob_sentiment)
        xquik_df["svm_sentiment"] = svm_pipeline.predict(xquik_df["clean_tweet"])
        output_path = os.getenv("XQUIK_OUTPUT_PATH", "data/xquik_predictions.csv")
        xquik_df.to_csv(output_path, index=False)
        print(f"  Xquik predictions saved: {output_path}")

    # Best model
    best_name = max(ml_results, key=ml_results.get)
    best_pred = svm_pred if best_name == 'SVM' else nb_pred

    print(f"\n🏆 Best Model: {best_name} ({ml_results[best_name]:.3f})")
    print(f"\n{classification_report(y_test, best_pred, target_names=labels)}")

    # Plots
    plot_model_comparison(ml_results)
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plot_confusion_matrix(cm, best_name, labels)

    # Live prediction demo
    print("\n🔮 Live Prediction Demo:")
    test_tweets = [
        "This product is absolutely amazing! Love it!",
        "Worst service ever, completely disappointed",
        "The package arrived today as scheduled",
    ]
    for tweet in test_tweets:
        tb = get_textblob_sentiment(tweet)
        ml = svm_pipeline.predict([clean_tweet(tweet)])[0]
        print(f"  Tweet: {tweet[:50]}")
        print(f"  TextBlob: {tb} | SVM: {ml}\n")

    print("=" * 55)
    print(f"✅ COMPLETE — Best: {best_name} | Acc: {ml_results[best_name]:.3f}")
    print("=" * 55)
