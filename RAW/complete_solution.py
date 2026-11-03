"""
Smart Product Pricing Challenge - Complete Solution
Multimodal ensemble approach using text and image features
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Image Processing
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Text Processing
from sentence_transformers import SentenceTransformer

# Custom utilities
import sys
sys.path.append('src/')
from utils import download_images

# ==================== Configuration ====================
class Config:
    DATASET_FOLDER = 'dataset/'
    IMAGE_FOLDER = 'images/'
    RANDOM_STATE = 42
    N_FOLDS = 5
    
    # Feature dimensions
    TFIDF_MAX_FEATURES = 3000
    IMAGE_FEATURE_DIM = 512
    TEXT_EMBEDDING_DIM = 384
    
    # Model parameters
    XGB_PARAMS = {
        'max_depth': 7,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    LGB_PARAMS = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1
    }
    
    CAT_PARAMS = {
        'depth': 7,
        'learning_rate': 0.05,
        'iterations': 500,
        'random_state': RANDOM_STATE,
        'verbose': False
    }

# ==================== Text Feature Extractor ====================
class TextFeatureExtractor:
    def __init__(self, max_features=3000):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            strip_accents='unicode'
        )
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_statistical_features(self, text):
        """Extract basic text statistics"""
        if pd.isna(text):
            return np.zeros(10)
        
        text = str(text)
        words = text.split()
        
        features = [
            len(text),  # character count
            len(words),  # word count
            len(set(words)),  # unique words
            np.mean([len(w) for w in words]) if words else 0,  # avg word length
            text.count(','),  # comma count
            text.count('.'),  # period count
            sum(c.isdigit() for c in text),  # digit count
            sum(c.isupper() for c in text),  # uppercase count
            len(re.findall(r'\d+', text)),  # number count
            len(re.findall(r'pack|count|quantity|pcs|pieces', text.lower()))  # quantity indicators
        ]
        return np.array(features)
    
    def extract_price_indicators(self, text):
        """Extract features related to pricing hints"""
        if pd.isna(text):
            return np.zeros(5)
        
        text = str(text).lower()
        
        features = [
            int('premium' in text or 'luxury' in text),
            int('budget' in text or 'value' in text or 'economy' in text),
            int('pack' in text or 'bundle' in text),
            len(re.findall(r'\d+\s*(?:ml|kg|g|oz|lb|l)', text)),  # units
            len(re.findall(r'\d+', text))  # all numbers
        ]
        return np.array(features)
    
    def fit_transform_tfidf(self, texts):
        """Fit and transform TF-IDF features"""
        texts = texts.fillna('').astype(str)
        return self.tfidf.fit_transform(texts).toarray()
    
    def transform_tfidf(self, texts):
        """Transform TF-IDF features"""
        texts = texts.fillna('').astype(str)
        return self.tfidf.transform(texts).toarray()
    
    def get_embeddings(self, texts):
        """Get sentence embeddings"""
        texts = texts.fillna('').astype(str).tolist()
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def extract_all_features(self, df, fit=True):
        """Extract all text features"""
        print("Extracting text features...")
        
        # TF-IDF features
        if fit:
            tfidf_features = self.fit_transform_tfidf(df['catalog_content'])
        else:
            tfidf_features = self.transform_tfidf(df['catalog_content'])
        
        # Statistical features
        stat_features = np.array([
            self.extract_statistical_features(text) 
            for text in df['catalog_content']
        ])
        
        # Price indicators
        price_features = np.array([
            self.extract_price_indicators(text) 
            for text in df['catalog_content']
        ])
        
        # Sentence embeddings
        embeddings = self.get_embeddings(df['catalog_content'])
        
        # Combine all features
        all_features = np.hstack([
            tfidf_features,
            stat_features,
            price_features,
            embeddings
        ])
        
        print(f"Text features shape: {all_features.shape}")
        return all_features

# ==================== Image Feature Extractor ====================
class ImageFeatureExtractor:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_cnn_features(self, image_path):
        """Extract CNN features from image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(512)
    
    def extract_color_features(self, image_path):
        """Extract color histogram features"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((100, 100))
            img_array = np.array(img)
            
            # RGB histograms
            hist_r = np.histogram(img_array[:,:,0], bins=16, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=16, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=16, range=(0, 256))[0]
            
            # Normalize
            hist_features = np.concatenate([hist_r, hist_g, hist_b]) / (100 * 100)
            
            # Basic stats
            mean_colors = np.mean(img_array, axis=(0, 1))
            std_colors = np.std(img_array, axis=(0, 1))
            
            return np.concatenate([hist_features, mean_colors, std_colors])
        except:
            return np.zeros(48 + 6)
    
    def extract_image_features(self, image_link):
        """Extract all image features for a single image"""
        filename = Path(image_link).name
        image_path = os.path.join(self.image_folder, filename)
        
        if not os.path.exists(image_path):
            # Return zero features if image doesn't exist
            return np.zeros(512 + 54)
        
        cnn_features = self.extract_cnn_features(image_path)
        color_features = self.extract_color_features(image_path)
        
        return np.concatenate([cnn_features, color_features])
    
    def extract_all_features(self, df):
        """Extract features for all images"""
        print("Extracting image features...")
        features = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing image {idx}/{len(df)}")
            
            img_features = self.extract_image_features(row['image_link'])
            features.append(img_features)
        
        features = np.array(features)
        print(f"Image features shape: {features.shape}")
        return features

# ==================== Ensemble Model ====================
class EnsembleModel:
    def __init__(self, config):
        self.config = config
        self.models = {
            'xgb': [],
            'lgb': [],
            'cat': []
        }
        self.scaler = StandardScaler()
        
    def calculate_smape(self, y_true, y_pred):
        """Calculate SMAPE score"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        smape = np.mean(diff / denominator) * 100
        return smape
    
    def train_fold(self, X_train, y_train, X_val, y_val):
        """Train models for one fold"""
        fold_models = {}
        fold_scores = {}
        
        # XGBoost
        print("  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50,
                     verbose=False)
        xgb_pred = xgb_model.predict(X_val)
        xgb_score = self.calculate_smape(y_val, xgb_pred)
        fold_models['xgb'] = xgb_model
        fold_scores['xgb'] = xgb_score
        
        # LightGBM
        print("  Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(**self.config.LGB_PARAMS)
        lgb_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50)])
        lgb_pred = lgb_model.predict(X_val)
        lgb_score = self.calculate_smape(y_val, lgb_pred)
        fold_models['lgb'] = lgb_model
        fold_scores['lgb'] = lgb_score
        
        # CatBoost
        print("  Training CatBoost...")
        cat_model = CatBoostRegressor(**self.config.CAT_PARAMS)
        cat_model.fit(X_train, y_train,
                     eval_set=(X_val, y_val),
                     early_stopping_rounds=50,
                     verbose=False)
        cat_pred = cat_model.predict(X_val)
        cat_score = self.calculate_smape(y_val, cat_pred)
        fold_models['cat'] = cat_model
        fold_scores['cat'] = cat_score
        
        print(f"  Fold scores - XGB: {xgb_score:.2f}%, LGB: {lgb_score:.2f}%, CAT: {cat_score:.2f}%")
        
        return fold_models, fold_scores
    
    def train(self, X, y):
        """Train ensemble with cross-validation"""
        print("\nTraining ensemble models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Fold cross-validation
        kfold = KFold(n_splits=self.config.N_FOLDS, shuffle=True, 
                      random_state=self.config.RANDOM_STATE)
        
        all_scores = {'xgb': [], 'lgb': [], 'cat': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            print(f"\nFold {fold + 1}/{self.config.N_FOLDS}")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            fold_models, fold_scores = self.train_fold(X_train, y_train, X_val, y_val)
            
            # Store models and scores
            for model_name in ['xgb', 'lgb', 'cat']:
                self.models[model_name].append(fold_models[model