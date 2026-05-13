# IMDb Sentiment Analysis
### MATH 373: Introduction to Statistical Learning — Final Project
**Authors:** Disha Khati, Kailash Turimella, Tyler Soong

---

## Overview
This project applies natural language processing and classical machine learning to predict the sentiment (positive/negative) of IMDb movie reviews. We combine two feature extraction approaches — TF-IDF (lexical) and Latent Dirichlet Allocation (semantic topic modeling) — across multiple classifier families to compare their predictive power and interpretability.

---

## Dataset
- **Source:** [IMDb Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)
- 50,000 reviews, perfectly balanced: 25,000 positive / 25,000 negative
- Download the CSV and place it in the project root as `IMDBDataset.csv`

---

## Project Structure
final-project/
├── Final_project.ipynb       # Main Jupyter notebook — full pipeline
├── IMDBDataset.csv           # Dataset (download from Kaggle)
├── QPDAC_Report.docx         # Written report with analysis and findings
└── README.md

---

## Pipeline Summary

### 1. Text Cleaning
- Strip HTML tags (`<br />` artifacts from web scraping)
- Remove punctuation and numbers
- Lowercase
- Remove stopwords (NLTK English stopword list)
- Lemmatize using `WordNetLemmatizer`

### 2. Feature Engineering
| Feature Set | Method | Shape |
|---|---|---|
| TF-IDF | `TfidfVectorizer` (unigrams + bigrams, max 10k features) | 40,000 × 10,000 |
| LDA topics | `CountVectorizer` → `LatentDirichletAllocation` (10 topics) | 40,000 × 10 |
| Combined | `hstack(TF-IDF, LDA)` | 40,000 × 10,010 |

### 3. Chi-squared Feature Selection
Chi-squared testing identifies the most sentiment-discriminative words across the training set. Top words include "bad", "worst", "waste", "awful", and "great" — confirming that strongly evaluative language drives classification.

### 4. Classifiers
| Model | Features | Rationale |
|---|---|---|
| `ComplementNB` | TF-IDF | Handles float features; uses complement class estimation |
| `MultinomialNB` | Raw counts | Designed for integer word frequencies |
| `GaussianNB` | LDA proportions | Handles continuous float features |
| `RandomForestClassifier` | All three | No distributional assumptions; isolates feature effect |

---

## Results

| Model | Accuracy |
|---|---|
| TF-IDF + ComplementNB | **86.51%** |
| TF-IDF + LDA Combined + RF | **86.44%** |
| TF-IDF + LDA Combined (NB) | 86.07% |
| Counts + MultinomialNB | 85.78% |
| TF-IDF + Random Forest | 85.61% |
| LDA + Random Forest | 81.82% |
| LDA + GaussianNB | 79.51% |

**Key findings:**
- TF-IDF + ComplementNB is the best single-feature model — simple methods win
- Random Forest exploits combined TF-IDF + LDA features better than Naive Bayes
- Reducing LDA topics from 15 → 10 improved interpretability and accuracy (77.35% → 79.51%)
- LDA's primary value is interpretability — discovered topics include horror, comedy, musicals, war films, and TV shows

---

## Requirements
python >= 3.10
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
scipy

Install dependencies:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn nltk scipy
```

Download NLTK data (run once):
```python
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
```

---

## How to Run
1. Clone the repo and navigate to the project folder
2. Download `IMDBDataset.csv` from Kaggle and place it in the project root
3. Open `Final_project.ipynb` in Jupyter or VS Code
4. Run all cells top to bottom (`Kernel → Restart & Run All`)

> ⚠️ LDA fitting on 40,000 reviews takes several minutes. Random Forest on TF-IDF features may take 10–20 minutes depending on your machine.

---

## Future Work
- **Aspect-based sentiment:** identify which film aspect (acting, plot, music) the sentiment applies to using LDA topics
- **Hyperparameter tuning:** grid search on Random Forest (`n_estimators`, `max_depth`) and LDA topic count
- **Boosting:** XGBoost or AdaBoost on TF-IDF features
- **Transformers:** BERT/RoBERTa fine-tuning would likely push accuracy above 90%
- **Letterboxd:** apply pipeline to Letterboxd-specific review data — more informal, ironic tone

---

## Acknowledgements
Dataset courtesy of [Lakshmi Narayana](https://www.kaggle.com/lakshmi25npathi) on Kaggle.
Course: MATH 373 — Introduction to Statistical Learning, University of San Francisco.