- Learns optimal weights automatically
- Prevents overfitting through CV
```

### Hyperparameter Choices (Research-Optimized)

```python
# Learning Rate: 0.05
# Research: Sweet spot between speed and accuracy
# Too high (>0.1): Unstable, overfits
# Too low (<0.01): Too slow, underfits
learning_rate = 0.05

# Tree Depth: 7
# Research: Prevents overfitting on 75K samples
# Deeper (>10): Overfits on training data
# Shallower (<5): Underfits, misses patterns
max_depth = 7

# Estimators: 1000 with early stopping
# Research: Let early stopping find optimal
# Will typically stop around 200-400 iterations
n_estimators = 1000
early_stopping_rounds = 50

# Subsampling: 0.8
# Research: Reduces overfitting, adds randomness
# Trains on 80% of data each iteration
subsample = 0.8
colsample_bytree = 0.8

# Regularization
# Research: L1+L2 prevents overfitting
reg_alpha = 0.1   # L1 (Lasso)
reg_lambda = 1.0  # L2 (Ridge)
```

---

## 🚀 Performance Optimization Strategies

### 1. Stratified Cross-Validation

```python
# Why: Price distribution is skewed
# Solution: Stratify by price quantiles

price_bins = pd.qcut(prices, q=10, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True)

# Ensures each fold has:
# - Similar price distribution
# - Representative of test set
# - More reliable CV scores
```

### 2. Feature Scaling

```python
# RobustScaler vs StandardScaler
scaler = RobustScaler()  # Better for outliers

# Why RobustScaler:
# - Uses median and IQR (not mean/std)
# - Robust to price outliers ($500+ products)
# - Preserves distribution shape
```

### 3. Early Stopping

```python
# Prevents overfitting
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)

# Stops when validation loss doesn't improve for 50 rounds
# Saves training time
# Prevents overfitting
```

### 4. Image Processing Optimization

```python
# Batch processing for speed
batch_size = 32
dataloader = DataLoader(images, batch_size=batch_size)

# GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Expected speedup:
# CPU: ~2-3 images/second
# GPU: ~20-30 images/second
```

### 5. Missing Data Handling

```python
# Graceful degradation for missing images
if not os.path.exists(image_path):
    return np.zeros(feature_dim)  # Zero padding

# Why it works:
# - Text features carry most information
# - Image features are additive bonus
# - Models learn to handle zero vectors
```

---

## 📁 Complete File Structure

```
smart-pricing-challenge/
│
├── dataset/
│   ├── train.csv                 # 75K training samples
│   ├── test.csv                  # 75K test samples
│   ├── sample_test.csv
│   ├── sample_test_out.csv
│   └── test_out.csv              # YOUR OUTPUT (generated)
│
├── images/                        # Downloaded images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── models/                        # Saved models (generated)
│   ├── base_models.pkl
│   ├── scaler.pkl
│   └── stacking_model.pkl
│
├── src/
│   └── utils.py                  # Image download utilities
│
├── advanced_pricing_solution.py  # MAIN CODE (11-13% SMAPE target)
├── complete_pricing_solution.py  # Earlier version (13-15% target)
├── simplified_pricing_solution.py # Quick baseline (16-18% target)
│
├── requirements.txt              # Dependencies
├── QUICKSTART_GUIDE.md          # Getting started
├── Solution_Documentation.md     # Submission document
├── COMPREHENSIVE_PROJECT_GUIDE.md # This file
│
└── notebooks/                    # Optional analysis
    ├── eda.ipynb                 # Exploratory analysis
    ├── feature_analysis.ipynb    # Feature importance
    └── model_comparison.ipynb    # Model benchmarking
```

---

## 🎓 Best Practices from Research

### 1. Feature Engineering Principles

```python
# DO: Create interpretable features
quantity = extract_pack_quantity(text)  # Clear meaning
brand_present = detect_brand(text)      # Business relevant

# DON'T: Create too many correlated features
# Avoid: char_count, word_count×avg_word_length (redundant)

# DO: Normalize features appropriately
price_per_unit = price / quantity      # Comparable across products
log_price = np.log(price + 1)         # Handle skewness

# DON'T: Use raw pixel values
# Use: CNN embeddings instead
```

### 2. Model Training Principles

```python
# DO: Use cross-validation
scores = []
for fold in kfold.split(X, y):
    score = train_and_evaluate(fold)
    scores.append(score)
print(f"CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# DON'T: Trust single train/val split
# Can be misleading due to random split luck

# DO: Monitor multiple metrics
metrics = {
    'SMAPE': calculate_smape(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'R²': r2_score(y_true, y_pred)
}

# DON'T: Focus only on SMAPE
# Other metrics provide complementary insights
```

### 3. Ensemble Principles

```python
# DO: Combine diverse models
ensemble = [
    xgboost,      # Gradient boosting
    lightgbm,     # Histogram-based
    catboost,     # Ordered boosting  
    random_forest # Bagging
]

# DON'T: Ensemble similar models
# bad_ensemble = [xgb1, xgb2, xgb3]  # All same algorithm

# DO: Weight by validation performance
weights = {
    'xgb': 0.35,  # Best performer
    'lgb': 0.35,  # Second best
    'cat': 0.25,  # Third
    'rf': 0.05    # Adds diversity
}

# DON'T: Use equal weights blindly
# Some models perform better than others
```

### 4. Validation Principles

```python
# DO: Stratified sampling for regression
bins = pd.qcut(y, q=10, labels=False)
skf = StratifiedKFold(n_splits=5)
for train_idx, val_idx in skf.split(X, bins):
    # Train and validate

# DON'T: Random split for skewed targets
# Can create unrepresentative folds

# DO: Check for data leakage
# Fit on train only, transform on train+val/test
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# DON'T: Fit on all data
# scaler.fit(X_all)  # WRONG! Data leakage
```

---

## 🔬 Advanced Techniques (Optional Improvements)

### 1. Hyperparameter Optimization with Optuna

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# Expected improvement: 0.5-1% SMAPE
```

### 2. Target Encoding for Categorical Features

```python
# Create category from text patterns
def extract_category(text):
    if 'electronic' in text.lower(): return 'electronics'
    if 'food' in text.lower(): return 'food'
    # ... more categories
    return 'other'

df['category'] = df['catalog_content'].apply(extract_category)

# Target encode
from category_encoders import TargetEncoder
encoder = TargetEncoder()
df['category_encoded'] = encoder.fit_transform(df['category'], df['price'])

# Expected improvement: 0.3-0.5% SMAPE
```

### 3. Price Range-Specific Models

```python
# Train separate models for different price ranges
low_model = train_model(df[df['price'] < 20])    # Budget products
mid_model = train_model(df[(df['price'] >= 20) & (df['price'] < 100)])
high_model = train_model(df[df['price'] >= 100])  # Premium products

# Predict based on predicted range (2-stage)
def predict_with_ranges(features):
    # Stage 1: Predict price range
    range_pred = range_classifier.predict(features)
    
    # Stage 2: Use appropriate model
    if range_pred == 'low':
        return low_model.predict(features)
    elif range_pred == 'mid':
        return mid_model.predict(features)
    else:
        return high_model.predict(features)

# Expected improvement: 1-2% SMAPE
```

### 4. CLIP for Vision-Language Features

```python
import torch
import clip

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_clip_features(image_path, text):
    # Process image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Process text
    text_tokens = clip.tokenize([text]).to(device)
    
    # Get features
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
    
    # Combine (visual-semantic alignment)
    combined = torch.cat([image_features, text_features], dim=1)
    return combined.cpu().numpy()

# Expected improvement: 0.5-1.5% SMAPE
# Note: CLIP is 151M params, within 8B limit
```

### 5. Pseudo-Labeling (Semi-Supervised)

```python
# Step 1: Train on labeled train data
model.fit(X_train, y_train)

# Step 2: Predict on test data (high confidence only)
test_preds = model.predict(X_test)
confidence = model.predict_proba(X_test)  # If available

# Step 3: Add high-confidence predictions to training
high_conf_mask = confidence > 0.9
X_pseudo = X_test[high_conf_mask]
y_pseudo = test_preds[high_conf_# Smart Product Pricing Challenge 2025 - Complete Project Guide

## 📚 Table of Contents
1. [Project Overview](#project-overview)
2. [Research Insights from PDF](#research-insights)
3. [Solution Architecture](#solution-architecture)
4. [Complete Code Files](#complete-code-files)
5. [Feature Engineering Deep Dive](#feature-engineering)
6. [Model Selection & Rationale](#model-selection)
7. [Performance Optimization](#performance-optimization)
8. [Deployment Strategy](#deployment-strategy)
9. [Best Practices & Tips](#best-practices)

---

## 🎯 Project Overview

### Challenge Details
- **Dataset Size:** 75,000 training + 75,000 test samples
- **Modalities:** Text (catalog_content) + Images (product images)
- **Target:** Product price prediction (positive floats)
- **Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
- **Target SMAPE:** 11-13% (based on research)
- **Model Constraint:** MIT/Apache 2.0 license, ≤8B parameters

### Key Challenge Rules
⚠️ **STRICTLY PROHIBITED:**
- External price lookup from web/APIs
- Web scraping product prices
- Using external pricing databases

✅ **ALLOWED:**
- Pre-trained models (ResNet, BERT, CLIP)
- Feature engineering from provided data
- Ensemble methods
- Cross-validation techniques

---

## 📊 Research Insights from PDF

### Expected Performance Benchmarks

Based on the comprehensive research document:

| Approach | Expected SMAPE | Key Insight |
|----------|----------------|-------------|
| Text Only (TF-IDF) | 18-20% | Baseline performance |
| Text + Embeddings | 15-17% | BERT/Sentence-BERT helps |
| Images Only (CNN) | 16-18% | Visual features alone |
| Text + Images (Early Fusion) | 13-15% | Combined features |
| **Ensemble Multimodal** | **11-13%** | **Best performance** |
| Stacking Regressor | 11-12% | Meta-learning optimal |

### Key Research Findings

1. **Multimodal AI Market Growth:** $55.54B by 2035 (37.2% CAGR)
2. **Ensemble Methods:** Consistently outperform single models by 2-4%
3. **Gradient Boosting Dominance:** XGBoost, LightGBM, CatBoost are top performers
4. **Feature Engineering Impact:** 30-40% performance improvement with proper features
5. **Cross-Validation Critical:** Stratified sampling prevents overfitting

### Recommended Tech Stack (from Research)

**Text Processing:**
- TF-IDF (5000 features recommended)
- BERT embeddings for contextual understanding
- Statistical features (word count, sentence count)
- Domain-specific parsing (quantities, units, brands)

**Image Processing:**
- EfficientNet (best accuracy/efficiency trade-off)
- ResNet-18/50 (proven performance)
- CLIP (visual-semantic alignment)
- Color statistics, aspect ratio, brightness

**Ensemble Models:**
- XGBoost (L1/L2 regularization)
- LightGBM (histogram-based, faster)
- CatBoost (excellent for categorical)
- Random Forest (robust baseline)
- Stacking Regressor (meta-learner)

---

## 🏗️ Solution Architecture

### Multi-Level Architecture (Based on Research)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT DATA LAYER                         │
│  75K Training Samples + 75K Test Samples                     │
│  • catalog_content (text)                                    │
│  • image_link (URL)                                          │
│  • price (target - train only)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION LAYER                    │
├──────────────────────────┬──────────────────────────────────┤
│   TEXT PROCESSING        │    IMAGE PROCESSING              │
├──────────────────────────┼──────────────────────────────────┤
│ 1. TF-IDF (5000 dims)    │ 1. CNN Features (512-2048 dims) │
│    • Unigrams + Bigrams  │    • EfficientNet-B0/B4          │
│    • min_df=3, max_df=0.9│    • ResNet-18/50                │
│                          │    • CLIP embeddings (optional)  │
│ 2. BERT Embeddings       │                                  │
│    • 384/768 dimensions  │ 2. Color Features (54 dims)     │
│    • all-MiniLM-L6-v2    │    • RGB histograms              │
│    • Contextual meaning  │    • Dominant colors             │
│                          │    • Brightness/Contrast         │
│ 3. Statistical (20 dims) │                                  │
│    • Word/char counts    │ 3. Structural (10 dims)         │
│    • Punctuation density │    • Dimensions, aspect ratio    │
│    • Number occurrences  │    • File size indicators        │
│                          │    • Image quality metrics       │
│ 4. Domain-Specific       │                                  │
│    • Quantity extraction │ 4. Advanced (Optional)          │
│    • Unit standardization│    • Object detection features   │
│    • Brand mentions      │    • OCR text extraction         │
│    • Quality indicators  │    • Product segmentation        │
└──────────────────────────┴──────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE FUSION LAYER                      │
│  Concatenate: ~5000-8000 total dimensions                   │
│  • Text features: ~5400 dims                                 │
│  • Image features: ~600 dims                                 │
│  • Standard Scaling applied                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 ENSEMBLE MODELING LAYER                      │
├─────────────┬─────────────┬─────────────┬─────────────┐
│  XGBoost    │  LightGBM   │  CatBoost   │Random Forest│
│  (35% wt)   │  (35% wt)   │  (25% wt)   │  (5% wt)    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ 5-Fold CV   │ 5-Fold CV   │ 5-Fold CV   │ 5-Fold CV   │
│ Early Stop  │ Early Stop  │ Early Stop  │ n_est=500   │
│ max_depth=7 │ num_leaf=31 │ depth=7     │ max_depth=15│
│ lr=0.05     │ lr=0.05     │ lr=0.05     │ min_split=5 │
└─────────────┴─────────────┴─────────────┴─────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              META-LEARNER (STACKING) - OPTIONAL              │
│  Ridge/Lasso Regression on base model predictions           │
│  • Learn optimal weights automatically                       │
│  • Further 0.5-1% SMAPE improvement                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT PREDICTION LAYER                    │
│  Weighted Average → Ensure Positive → Round to 2 decimals   │
│  Expected SMAPE: 11-13%                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Complete Code Files

### File 1: `advanced_pricing_solution.py` (Enhanced Based on Research)

```python
"""
Advanced Smart Product Pricing Solution
Based on research paper recommendations for achieving 11-13% SMAPE
Implements: Multimodal features + Ensemble + Stacking + Advanced techniques
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# NLP
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Utils
from tqdm import tqdm
import joblib
import json

# Custom utils
import sys
sys.path.append('src/')
from utils import download_images


# ============================================================================
#                           CONFIGURATION
# ============================================================================

class AdvancedConfig:
    """Configuration based on research recommendations"""
    
    # Paths
    DATASET_FOLDER = 'dataset/'
    IMAGE_FOLDER = 'images/'
    MODEL_FOLDER = 'models/'
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Cross-validation (Research recommendation: stratified by price quantiles)
    N_FOLDS = 5
    STRATIFY_BINS = 10  # Stratify by price deciles
    
    # Text Features (Research: 5000 TF-IDF optimal)
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 3
    TFIDF_MAX_DF = 0.9
    
    # Sentence Embeddings
    SENTENCE_MODEL = 'all-MiniLM-L6-v2'  # 384 dims, fast
    # Alternative: 'all-mpnet-base-v2'  # 768 dims, better quality
    
    # Image Features (Research: EfficientNet best for accuracy/speed)
    IMAGE_MODEL = 'efficientnet_b0'  # 1280 dims
    # Alternative: 'resnet50' (2048 dims), 'efficientnet_b4' (1792 dims)
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    
    # Model Parameters (Tuned based on research)
    XGB_PARAMS = {
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    LGB_PARAMS = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
    
    CAT_PARAMS = {
        'depth': 7,
        'learning_rate': 0.05,
        'iterations': 1000,
        'l2_leaf_reg': 3,
        'random_state': RANDOM_STATE,
        'verbose': False
    }
    
    RF_PARAMS = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Ensemble weights (can be optimized)
    ENSEMBLE_WEIGHTS = {
        'xgb': 0.35,
        'lgb': 0.35,
        'cat': 0.25,
        'rf': 0.05
    }
    
    # Stacking meta-learner
    USE_STACKING = True  # Research shows 0.5-1% improvement
    STACKING_CV = 3


# ============================================================================
#                        ADVANCED TEXT FEATURES
# ============================================================================

class AdvancedTextExtractor:
    """Enhanced text feature extraction based on research"""
    
    def __init__(self, config):
        self.config = config
        
        # TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        # Sentence embeddings
        self.sentence_model = SentenceTransformer(config.SENTENCE_MODEL)
        
        # Brand patterns (common e-commerce brands)
        self.brand_patterns = [
            r'\b(amazon|apple|samsung|sony|lg|dell|hp|lenovo|asus)\b',
            r'\b(nike|adidas|puma|reebok|under armour)\b',
            r'\b(nestle|kraft|pepsi|coca cola|unilever)\b',
            r'\b(loreal|maybelline|revlon|nivea|dove)\b'
        ]
        
    def extract_statistical_features(self, text):
        """Extract comprehensive statistical features"""
        if pd.isna(text):
            return np.zeros(25)
        
        text_str = str(text)
        text_lower = text_str.lower()
        words = text_str.split()
        sentences = text_str.split('.')
        
        features = [
            # Basic counts
            len(text_str),  # character count
            len(words),  # word count
            len(set(words)),  # unique words
            len(sentences),  # sentence count
            
            # Average metrics
            np.mean([len(w) for w in words]) if words else 0,  # avg word length
            np.std([len(w) for w in words]) if len(words) > 1 else 0,  # std word length
            len(text_str) / len(words) if words else 0,  # chars per word
            len(words) / len(sentences) if sentences else 0,  # words per sentence
            
            # Punctuation
            text_str.count(','),
            text_str.count('.'),
            text_str.count(':'),
            text_str.count(';'),
            text_str.count('!'),
            text_str.count('?'),
            
            # Character types
            sum(c.isdigit() for c in text_str),  # digits
            sum(c.isupper() for c in text_str),  # uppercase
            sum(c.islower() for c in text_str),  # lowercase
            sum(c.isspace() for c in text_str),  # spaces
            
            # Special ratios
            sum(c.isdigit() for c in text_str) / len(text_str) if text_str else 0,
            sum(c.isupper() for c in text_str) / len(text_str) if text_str else 0,
            
            # Numbers and units
            len(re.findall(r'\d+', text_str)),  # number occurrences
            len(re.findall(r'\d+\.\d+', text_str)),  # decimal numbers
            len(re.findall(r'\d+\s*(?:ml|kg|g|oz|lb|l|mg|kg|liter)', text_lower)),
            
            # Lexical diversity
            len(set(words)) / len(words) if words else 0,  # unique word ratio
            
            # Price-related word count
            sum(1 for w in words if w.lower() in ['pack', 'bundle', 'set', 'combo', 'value'])
        ]
        
        return np.array(features)
    
    def extract_domain_features(self, text):
        """Extract domain-specific pricing features"""
        if pd.isna(text):
            return np.zeros(20)
        
        text_str = str(text).lower()
        
        features = [
            # Quantity indicators
            int('pack' in text_str),
            int('bundle' in text_str),
            int('set' in text_str),
            int('combo' in text_str),
            int('value pack' in text_str),
            int('bulk' in text_str),
            
            # Quality indicators
            int('premium' in text_str),
            int('luxury' in text_str),
            int('professional' in text_str),
            int('pro' in text_str),
            int('deluxe' in text_str),
            int('ultimate' in text_str),
            
            # Budget indicators
            int('budget' in text_str),
            int('economy' in text_str),
            int('value' in text_str),
            int('basic' in text_str),
            
            # Organic/Natural
            int('organic' in text_str),
            int('natural' in text_str),
            int('eco' in text_str or 'eco-' in text_str),
            
            # Brand presence (any brand match)
            int(any(re.search(pattern, text_str) for pattern in self.brand_patterns))
        ]
        
        return np.array(features)
    
    def extract_quantity_value(self, text):
        """Extract numerical quantity from text (e.g., 'Pack of 12' → 12)"""
        if pd.isna(text):
            return 1.0
        
        text_str = str(text).lower()
        
        # Pattern: "pack of X", "X pack", "X count", etc.
        patterns = [
            r'pack\s+of\s+(\d+)',
            r'(\d+)\s+pack',
            r'(\d+)\s+count',
            r'(\d+)\s+pieces',
            r'(\d+)\s+pcs',
            r'count[:\s]+(\d+)',
            r'quantity[:\s]+(\d+)',
            r'(\d+)\s*x\s*\d+',  # "12x500ml"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_str)
            if match:
                return float(match.group(1))
        
        return 1.0
    
    def extract_unit_size(self, text):
        """Extract unit size (e.g., '500ml' → 500)"""
        if pd.isna(text):
            return 0.0
        
        text_str = str(text).lower()
        
        # Pattern: number followed by unit
        patterns = [
            r'(\d+\.?\d*)\s*(ml|milliliter)',
            r'(\d+\.?\d*)\s*(l|liter)',
            r'(\d+\.?\d*)\s*(g|gram)',
            r'(\d+\.?\d*)\s*(kg|kilogram)',
            r'(\d+\.?\d*)\s*(oz|ounce)',
            r'(\d+\.?\d*)\s*(lb|pound)',
        ]
        
        # Conversion to standard unit (ml/g)
        conversions = {
            'ml': 1, 'milliliter': 1,
            'l': 1000, 'liter': 1000,
            'g': 1, 'gram': 1,
            'kg': 1000, 'kilogram': 1000,
            'oz': 28.35, 'ounce': 28.35,
            'lb': 453.59, 'pound': 453.59
        }
        
        for pattern in patterns:
            match = re.search(pattern, text_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                return value * conversions.get(unit, 1)
        
        return 0.0
    
    def fit_transform(self, df):
        """Fit and transform all text features"""
        print("Extracting text features...")
        
        texts = df['catalog_content'].fillna('').astype(str)
        
        # TF-IDF
        print("  - TF-IDF vectorization...")
        tfidf_features = self.tfidf.fit_transform(texts).toarray()
        
        # Statistical features
        print("  - Statistical features...")
        stat_features = np.array([
            self.extract_statistical_features(text) for text in tqdm(texts, desc="Stats")
        ])
        
        # Domain features
        print("  - Domain-specific features...")
        domain_features = np.array([
            self.extract_domain_features(text) for text in texts
        ])
        
        # Quantity and size
        print("  - Quantity extraction...")
        quantity_features = np.array([
            [self.extract_quantity_value(text), self.extract_unit_size(text)]
            for text in texts
        ])
        
        # Sentence embeddings
        print("  - Sentence embeddings...")
        embeddings = self.sentence_model.encode(
            texts.tolist(), 
            show_progress_bar=True,
            batch_size=64
        )
        
        # Combine all
        all_features = np.hstack([
            tfidf_features,          # 5000
            stat_features,           # 25
            domain_features,         # 20
            quantity_features,       # 2
            embeddings              # 384
        ])
        
        print(f"  Total text features: {all_features.shape}")
        return all_features
    
    def transform(self, df):
        """Transform text features (without fitting)"""
        texts = df['catalog_content'].fillna('').astype(str)
        
        tfidf_features = self.tfidf.transform(texts).toarray()
        stat_features = np.array([self.extract_statistical_features(text) for text in texts])
        domain_features = np.array([self.extract_domain_features(text) for text in texts])
        quantity_features = np.array([[self.extract_quantity_value(text), self.extract_unit_size(text)] for text in texts])
        embeddings = self.sentence_model.encode(texts.tolist(), show_progress_bar=True, batch_size=64)
        
        all_features = np.hstack([tfidf_features, stat_features, domain_features, quantity_features, embeddings])
        return all_features


# ============================================================================
#                        ADVANCED IMAGE FEATURES
# ============================================================================

class AdvancedImageExtractor:
    """Enhanced image feature extraction"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load EfficientNet-B0 (research recommendation)
        self.model = models.efficientnet_b0(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def extract_cnn_features(self, image_path):
        """Extract deep CNN features"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            # Return zero vector for failed images
            return np.zeros(1280)  # EfficientNet-B0 output
    
    def extract_color_features(self, image_path):
        """Extract comprehensive color features"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_resized = img.resize((100, 100))
            img_array = np.array(img_resized)
            
            # RGB histograms (16 bins each = 48 features)
            hist_r = np.histogram(img_array[:,:,0], bins=16, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=16, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=16, range=(0, 256))[0]
            hist_features = np.concatenate([hist_r, hist_g, hist_b]) / (100 * 100)
            
            # Color statistics
            mean_rgb = np.mean(img_array, axis=(0, 1))
            std_rgb = np.std(img_array, axis=(0, 1))
            min_rgb = np.min(img_array, axis=(0, 1))
            max_rgb = np.max(img_array, axis=(0, 1))
            
            # Brightness and contrast
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Dominant color (most common color cluster)
            pixels = img_array.reshape(-1, 3)
            dominant_color = np.median(pixels, axis=0)
            
            color_stats = np.concatenate([
                mean_rgb, std_rgb, min_rgb, max_rgb, 
                [brightness, contrast], dominant_color
            ])
            
            return np.concatenate([hist_features, color_stats])
        except:
            return np.zeros(48 + 17)
    
    def extract_structural_features(self, image_path):
        """Extract image structural features"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            features = [
                width,
                height,
                width / height if height > 0 else 1.0,  # aspect ratio
                width * height,  # total pixels
                np.log(width * height + 1),  # log pixels
                int(width > height),  # landscape
                int(width < height),  # portrait
                int(abs(width - height) < 50),  # square-ish
            ]
            
            # File size (approximate from dimensions)
            file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
            features.append(file_size / 1024)  # KB
            features.append(np.log(file_size + 1))
            
            return np.array(features)
        except:
            return np.zeros(10)
    
    def extract_all_features(self, df):
        """Extract all image features"""
        print("Extracting image features...")
        all_features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Images"):
            filename = Path(row['image_link']).name
            image_path = os.path.join(self.config.IMAGE_FOLDER, filename)
            
            if os.path.exists(image_path):
                cnn_feat = self.extract_cnn_features(image_path)
                color_feat = self.extract_color_features(image_path)
                struct_feat = self.extract_structural_features(image_path)
            else:
                # Missing image - zero features
                cnn_feat = np.zeros(1280)
                color_feat = np.zeros(65)
                struct_feat = np.zeros(10)
            
            combined = np.concatenate([cnn_feat, color_feat, struct_feat])
            all_features.append(combined)
        
        features = np.array(all_features)
        print(f"  Total image features: {features.shape}")
        return features


# ============================================================================
#                       ADVANCED ENSEMBLE MODEL
# ============================================================================

class AdvancedEnsemble:
    """Advanced ensemble with stacking"""
    
    def __init__(self, config):
        self.config = config
        self.models = {'xgb': [], 'lgb': [], 'cat': [], 'rf': []}
        self.stacking_model = None
        self.scaler = RobustScaler()  # More robust to outliers
        
    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        return np.mean(diff / (denominator + 1e-10)) * 100
    
    def create_price_bins(self, y, n_bins=10):
        """Create price bins for stratified sampling"""
        return pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
    
    def train_base_models(self, X, y):
        """Train base models with stratified K-fold"""
        print("\nTraining base models with stratified cross-validation...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create price bins for stratification
        price_bins = self.create_price_bins(y, self.config.STRATIFY_BINS)
        
        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.config.N_FOLDS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        
        all_scores = {'xgb': [], 'lgb': [], 'cat': [], 'rf': []}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, price_bins)):
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{self.config.N_FOLDS}")
            print(f"{'='*60}")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"Train: {X_train.shape}, Val: {X_val.shape}")
            print(f"Price range - Train: ${y_train.min():.2f}-${y_train.max():.2f}, Val: ${y_val.min():.2f}-${y_val.max():.2f}")
            
            # XGBoost
            print("\n  Training XGBoost...")
            xgb_model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            xgb_pred = xgb_model.predict(X_val)
            xgb_smape = self.calculate_smape(y_val, xgb_pred)
            self.models['xgb'].append(xgb_model)
            all_scores['xgb'].append(xgb_smape)
            print(f"    XGBoost SMAPE: {xgb_smape:.3f}%")
            
            # LightGBM
            print("  Training LightGBM...")
            lgb_model = lgb.LGBMRegressor(**self.config.LGB_PARAMS)
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
            lgb_pred = lgb_model.predict(X_val)
            lgb_smape = self.calculate_smape(y_val, lgb_pred)
            self.models['lgb'].append(lgb_model)
            all_scores['lgb'].append(lgb_smape)
            print(f"    LightGBM SMAPE: {lgb_smape:.3f}%")
            
            # CatBoost
            print("  Training CatBoost...")
            cat_model = CatBoostRegressor(**self.config.CAT_PARAMS)
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            cat_pred = cat_model.predict(X_val)
            cat_smape = self.calculate_smape(y_val, cat_pred)
            self.models['cat'].append(cat_model)
            all_scores['cat'].append(cat_smape)
            print(f"    CatBoost SMAPE: {cat_smape:.3f}%")
            
            # Random Forest
            print("  Training Random Forest...")
            rf_model = RandomForestRegressor(**self.config.RF_PARAMS)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_val)
            rf_smape = self.calculate_smape(y_val, rf_pred)
            self.models['rf'].append(rf_model)
            all_scores['rf'].append(rf_smape)
            print(f"    Random Forest SMAPE: {rf_smape:.3f}%")
            
            # Ensemble prediction for this fold
            ensemble_pred = (
                self.config.ENSEMBLE_WEIGHTS['xgb'] * xgb_pred +
                self.config.ENSEMBLE_WEIGHTS['lgb'] * lgb_pred +
                self.config.ENSEMBLE_WEIGHTS['cat'] * cat_pred +
                self.config.ENSEMBLE_WEIGHTS['rf'] * rf_pred
            )
            ensemble_smape = self.calculate_smape(y_val, ensemble_pred)
            print(f"\n  Fold {fold + 1} Ensemble SMAPE: {ensemble_smape:.3f}%")
        
        # Print final CV results
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        for model_name in ['xgb', 'lgb', 'cat', 'rf']:
            scores = all_scores[model_name]
            print(f"{model_name.upper():12s}: {np.mean(scores):.3f}% ± {np.std(scores):.3f}%")
        print("="*60)
    
    def train_stacking_model(self, X, y):
        """Train stacking meta-learner"""
        if not self.config.USE_STACKING:
            return
        
        print("\nTraining stacking meta-learner...")
        
        X_scaled = self.scaler.transform(X)
        
        # Create base estimators for stacking
        base_estimators = [
            ('xgb', xgb.XGBRegressor(**self.config.XGB_PARAMS)),
            ('lgb', lgb.LGBMRegressor(**self.config.LGB_PARAMS)),
            ('cat', CatBoostRegressor(**self.config.CAT_PARAMS)),
            ('rf', RandomForestRegressor(**self.config.RF_PARAMS))
        ]
        
        # Stacking with Ridge meta-learner
        self.stacking_model = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=self.config.STACKING_CV,
            n_jobs=-1
        )
        
        self.stacking_model.fit(X_scaled, y)
        print("  Stacking model trained successfully!")
    
    def predict(self, X, use_stacking=None):
        """Generate predictions"""
        if use_stacking is None:
            use_stacking = self.config.USE_STACKING
        
        X_scaled = self.scaler.transform(X)
        
        if use_stacking and self.stacking_model is not None:
            # Use stacking model
            predictions = self.stacking_model.predict(X_scaled)
        else:
            # Use weighted average of fold models
            all_preds = []
            
            for model_name, weight in self.config.ENSEMBLE_WEIGHTS.items():
                fold_preds = []
                for model in self.models[model_name]:
                    preds = model.predict(X_scaled)
                    fold_preds.append(preds)
                avg_preds = np.mean(fold_preds, axis=0)
                all_preds.append(weight * avg_preds)
            
            predictions = np.sum(all_preds, axis=0)
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0.01)
        return predictions
    
    def save_models(self, folder):
        """Save all trained models"""
        os.makedirs(folder, exist_ok=True)
        
        joblib.dump(self.models, os.path.join(folder, 'base_models.pkl'))
        joblib.dump(self.scaler, os.path.join(folder, 'scaler.pkl'))
        if self.stacking_model:
            joblib.dump(self.stacking_model, os.path.join(folder, 'stacking_model.pkl'))
        
        print(f"Models saved to {folder}")


# ============================================================================
#                           MAIN PIPELINE
# ============================================================================

class AdvancedPipeline:
    """Complete advanced pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.text_extractor = AdvancedTextExtractor(config)
        self.image_extractor = None
        self.ensemble = AdvancedEnsemble(config)
        
    def load_data(self):
        """Load datasets"""
        print("Loading datasets...")
        train_df = pd.read_csv(os.path.join(self.config.DATASET_FOLDER, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.config.DATASET_FOLDER, 'test.csv'))
        
        print(f"Train: {train_df.shape}")
        print(f"Test: {test_df.shape}")
        print(f"\nPrice statistics:")
        print(train_df['price'].describe())
        
        return train_df, test_df
    
    def download_images_if_needed(self, df, dataset_name):
        """Download images"""
        if not os.path.exists(self.config.IMAGE_FOLDER):
            os.makedirs(self.config.IMAGE_FOLDER)
        
        existing = len(os.listdir(self.config.IMAGE_FOLDER))
        print(f"\n{dataset_name} images: {existing} already downloaded")
        
        if existing < len(df) * 0.9:
            print(f"Downloading {dataset_name} images...")
            download_images(df['image_link'].tolist(), self.config.IMAGE_FOLDER)
    
    def extract_features(self, train_df, test_df, use_images=True):
        """Extract all features"""
        print("\n" + "="*60)
        print("FEATURE EXTRACTION PHASE")
        print("="*60)
        
        # Text features
        train_text = self.text_extractor.fit_transform(train_df)
        test_text = self.text_extractor.transform(test_df)
        
        if use_images:
            # Download images
            self.download_images_if_needed(train_df, "Training")
            self.download_images_if_needed(test_df, "Test")
            
            # Image features
            self.image_extractor = AdvancedImageExtractor(self.config)
            train_image = self.image_extractor.extract_all_features(train_df)
            test_image = self.image_extractor.extract_all_features(test_df)
            
            # Combine
            train_features = np.hstack([train_text, train_image])
            test_features = np.hstack([test_text, test_image])
        else:
            train_features = train_text
            test_features = test_text
        
        print(f"\nFinal features - Train: {train_features.shape}, Test: {test_features.shape}")
        return train_features, test_features
    
    def train(self, train_features, train_labels):
        """Train models"""
        print("\n" + "="*60)
        print("MODEL TRAINING PHASE")
        print("="*60)
        
        # Train base models
        self.ensemble.train_base_models(train_features, train_labels)
        
        # Train stacking model
        self.ensemble.train_stacking_model(train_features, train_labels)
        
        # Save models
        self.ensemble.save_models(self.config.MODEL_FOLDER)
    
    def predict(self, test_df, test_features):
        """Generate predictions"""
        print("\n" + "="*60)
        print("PREDICTION PHASE")
        print("="*60)
        
        predictions = self.ensemble.predict(test_features)
        
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': np.round(predictions, 2)
        })
        
        return output_df
    
    def run(self, use_images=True):
        """Run complete pipeline"""
        print("\n" + "="*60)
        print("ADVANCED SMART PRODUCT PRICING SOLUTION")
        print("Target SMAPE: 11-13% (Research-Based)")
        print("="*60)
        
        # Load data
        train_df, test_df = self.load_data()
        
        # Extract features
        train_features, test_features = self.extract_features(train_df, test_df, use_images)
        
        # Train
        self.train(train_features, train_df['price'].values)
        
        # Predict
        output_df = self.predict(test_df, test_features)
        
        # Save
        output_path = os.path.join(self.config.DATASET_FOLDER, 'test_out.csv')
        output_df.to_csv(output_path, index=False)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED!")
        print("="*60)
        print(f"\nOutput: {output_path}")
        print(f"Predictions: {len(output_df)}")
        print(f"\nPrice statistics:")
        print(output_df['price'].describe())
        
        return output_df


# ============================================================================
#                              MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_images', action='store_true', help='Skip images')
    parser.add_argument('--no_stacking', action='store_true', help='Skip stacking')
    args = parser.parse_args()
    
    # Configuration
    config = AdvancedConfig()
    if args.no_stacking:
        config.USE_STACKING = False
    
    # Set seeds
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)
    
    # Run pipeline
    pipeline = AdvancedPipeline(config)
    output_df = pipeline.run(use_images=not args.no_images)
    
    print("\n✓ Done! Submit test_out.csv")
```

---

## 🎯 Feature Engineering Deep Dive

### Text Feature Categories (Based on Research)

#### 1. TF-IDF Features (5000 dimensions)
```python
# Optimal configuration from research
TfidfVectorizer(
    max_features=5000,      # Research shows 5000 optimal
    ngram_range=(1, 2),     # Unigrams + Bigrams
    min_df=3,               # Remove rare terms
    max_df=0.9              # Remove too common terms
)

# Key captured patterns:
# - Brand names: "apple iphone", "samsung galaxy"
# - Product types: "wireless mouse", "organic honey"
# - Quality indicators: "premium quality", "professional grade"
```

#### 2. Statistical Features (25 dimensions)
```python
features = [
    # Basic counts
    character_count, word_count, unique_words, sentence_count,
    
    # Averages
    avg_word_length, std_word_length, chars_per_word, words_per_sentence,
    
    # Punctuation (indicates description quality)
    comma_count, period_count, colon_count, semicolon_count,
    
    # Character types
    digit_count, uppercase_count, lowercase_count,
    
    # Ratios (normalized features)
    digit_ratio, uppercase_ratio, lexical_diversity,
    
    # Numbers and units (strong price indicators)
    number_occurrences, decimal_numbers, unit_measurements
]
```

#### 3. Domain-Specific Features (20 dimensions)
```python
# Quantity indicators (multi-packs cost more)
pack_mention, bundle_mention, set_mention, combo_mention, value_pack, bulk

# Quality tier indicators
premium, luxury, professional, pro, deluxe, ultimate  # High-end
budget, economy, value, basic                          # Low-end

# Natural/Organic (premium pricing)
organic, natural, eco_friendly

# Brand presence (branded products cost more)
brand_detected
```

#### 4. Extracted Values (2 dimensions)
```python
# Quantity (e.g., "Pack of 12" → 12)
quantity_value = extract_from_patterns([
    'pack of X', 'X pack', 'X count', 'X pieces'
])

# Unit size (e.g., "500ml" → 500)
unit_size = extract_with_conversion({
    'ml': 1, 'l': 1000, 'g': 1, 'kg': 1000,
    'oz': 28.35, 'lb': 453.59
})
```

#### 5. Semantic Embeddings (384 dimensions)
```python
# Sentence-BERT captures semantic meaning
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)  # 384-dim vectors

# Similar products get similar embeddings
# "Organic Honey 500g" ≈ "Natural Bee Honey 16oz"
```

### Image Feature Categories

#### 1. CNN Deep Features (1280 dimensions)
```python
# EfficientNet-B0 (research recommendation)
model = efficientnet_b0(pretrained=True)
features = model(image)  # 1280-dim rich representation

# Captures:
# - Product packaging quality
# - Professional vs amateur photography
# - Single item vs multi-pack visualization
# - Product category (electronics, food, cosmetics)
```

#### 2. Color Features (65 dimensions)
```python
# RGB Histograms (16 bins × 3 channels = 48)
hist_r, hist_g, hist_b = compute_histograms(image, bins=16)

# Color statistics (17 features)
mean_rgb, std_rgb, min_rgb, max_rgb  # 12 features
brightness = np.mean(image)           # 1 feature
contrast = np.std(image)              # 1 feature
dominant_color = np.median(pixels)    # 3 features

# Why it matters:
# - Electronics: often black/grey/silver
# - Food: colorful packaging
# - Luxury: specific color palettes
```

#### 3. Structural Features (10 dimensions)
```python
width, height, aspect_ratio, total_pixels, log_pixels,
is_landscape, is_portrait, is_square,
file_size_kb, log_file_size

# Professional products often have:
# - Higher resolution
# - Square aspect ratio (1:1)
# - Larger file sizes
```

---

## 📈 Model Selection Rationale

### Why This Ensemble? (Research-Backed)

#### 1. XGBoost (Weight: 35%)
```python
Strengths:
- L1/L2 regularization prevents overfitting
- Excellent with high-dimensional features
- Handles missing values gracefully
- Fast prediction time

Research shows: Best for structured data with many features
```

#### 2. LightGBM (Weight: 35%)
```python
Strengths:
- Histogram-based algorithm (faster than XGBoost)
- Better with large datasets (75K samples)
- Lower memory footprint
- Comparable accuracy to XGBoost

Research shows: Often wins Kaggle competitions
```

#### 3. CatBoost (Weight: 25%)
```python
Strengths:
- Excellent handling of categorical features
- Ordered boosting reduces overfitting
- Built-in text feature support
- Robust to hyperparameter choices

Research shows: Good for diverse data types
```

#### 4. Random Forest (Weight: 5%)
```python
Strengths:
- Provides ensemble diversity
- Very robust (hard to overfit)
- Different learning paradigm than boosting
- Good feature importance

Research shows: Small weight but improves stability
```

#### 5. Stacking Meta-Learner (Optional)
```python
Ridge(alpha=1.0) on base model predictions

Research shows:
- Additional 0.5-1% SMAPE improvement
- Learns optimal weights automatically
- Prevents overfitting through CV