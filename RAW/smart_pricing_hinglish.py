#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 Smart Product Pricing Challenge - Prak Style Solution
===============================================
Bhai, ye solution IBM Watson jaisi advanced ML techniques use karta hai!
Multimodal approach hai - text aur images dono ko samjhata hai price predict karne ke liye

Author: Prakhar Sharma (prak05) - RIET B.Tech Student
Style: Technical par simple, just like AI internship projects! 💪
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')  # Console ko clean rakhne ke liye

# Ye sab ML libraries hai - sabse pehle import kar lete hai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb

# Image processing ke liye - CNN features extract karne ke liye
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Text embeddings ke liye - NLP magic! ✨
from sentence_transformers import SentenceTransformer

# Watson-style configuration class - clean aur organized!
class SmartPricingConfig:
    """Configuration parameters - sab settings ek jagah!"""
    
    # Folder structure - organized rakhte hai projects
    DATASET_FOLDER = 'dataset/'
    IMAGE_FOLDER = 'images/'
    
    # Model parameters - fine-tuned for best performance
    RANDOM_STATE = 42  # Answer to universe! 😄
    N_FOLDS = 3  # Cross validation folds - memory bachane ke liye kam rakha
    
    # Feature dimensions - balance performance aur speed
    TFIDF_MAX_FEATURES = 2000
    TEXT_EMBEDDING_DIM = 384
    
    # XGBoost parameters - ye model bohot accha hai price prediction ke liye
    XGB_PARAMS = {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1  # Sare CPU cores use karo bhai!
    }
    
    # LightGBM parameters - fast aur accurate
    LGB_PARAMS = {
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1  # Silent mode - console clean rakhte hai
    }

class TextFeatureEngineer:
    """
    Text features ko extract karta hai - bilkul Watson NLP jaisa!
    TF-IDF + Statistical features + Sentence embeddings = Power combo! 💯
    """
    
    def __init__(self, max_features=2000):
        print("🔤 Text Feature Engineer initialize ho raha hai...")
        
        # TF-IDF vectorizer - text ko numbers mein convert karta hai
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Single words + word pairs
            min_df=3,  # Kam se kam 3 documents mein hona chahiye
            max_df=0.9,  # 90% se zyada documents mein nahi hona chahiye
            strip_accents='unicode'  # Special characters handle karo
        )
        
        # Sentence transformer - advanced embeddings ke liye
        # IBM Watson jaisi NLP capabilities!
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_stats(self, text):
        """Basic text statistics - kitne words, chars etc"""
        if pd.isna(text):
            return np.zeros(8)  # Null handling
        
        text = str(text)
        words = text.split()
        
        # Statistical features - data science fundamentals
        features = [
            len(text),  # Character count
            len(words),  # Word count  
            len(set(words)),  # Unique words
            np.mean([len(w) for w in words]) if words else 0,  # Average word length
            text.count(','),  # Comma count - specification indicator
            sum(c.isdigit() for c in text),  # Digit count
            len(re.findall(r'\d+', text)),  # Number patterns
            len(re.findall(r'pack|quantity|pcs|pieces', text.lower()))  # Quantity words
        ]
        
        return np.array(features)
    
    def extract_price_hints(self, text):
        """Price related keywords detect karta hai - smart feature engineering!"""
        if pd.isna(text):
            return np.zeros(4)
        
        text = str(text).lower()
        
        # Price category indicators
        features = [
            int('premium' in text or 'luxury' in text),  # High-end products
            int('budget' in text or 'value' in text),    # Budget products  
            int('pack' in text or 'bundle' in text),     # Multi-pack items
            len(re.findall(r'\d+\s*(?:ml|kg|g|oz|lb)', text))  # Unit measurements
        ]
        
        return np.array(features)
    
    def process_text_features(self, df, is_training=True):
        """Complete text processing pipeline - ek function mein sab kuch!"""
        print("📝 Text features extract kar rahe hai...")
        
        # TF-IDF features - traditional NLP approach
        if is_training:
            tfidf_features = self.tfidf.fit_transform(df['catalog_content'].fillna('')).toarray()
        else:
            tfidf_features = self.tfidf.transform(df['catalog_content'].fillna('')).toarray()
        
        # Statistical features for each product
        stat_features = np.array([
            self.extract_text_stats(text) for text in df['catalog_content']
        ])
        
        # Price hint features  
        price_features = np.array([
            self.extract_price_hints(text) for text in df['catalog_content']
        ])
        
        # Sentence embeddings - advanced NLP magic!
        print("🧠 Sentence embeddings generate kar rahe hai...")
        embeddings = self.sentence_model.encode(
            df['catalog_content'].fillna('').astype(str).tolist(),
            show_progress_bar=True
        )
        
        # Combine all features - feature fusion technique
        all_features = np.hstack([
            tfidf_features,
            stat_features, 
            price_features,
            embeddings
        ])
        
        print(f"✅ Text features ready! Shape: {all_features.shape}")
        return all_features

class ImageFeatureEngineer:
    """
    Image se features extract karta hai - CNN power use karke!
    ResNet model use karte hai - pre-trained image recognition 🖼️
    """
    
    def __init__(self, image_folder):
        print("🖼️ Image Feature Engineer setup kar rahe hai...")
        
        self.image_folder = image_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device} - {'GPU speed!' if torch.cuda.is_available() else 'CPU mode'}")
        
        # Pre-trained ResNet model - ImageNet pe trained hai
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Last layer remove
        self.model = self.model.to(self.device)
        self.model.eval()  # Inference mode
        
        # Image preprocessing - standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_image_features(self, image_link):
        """Single image se features extract karta hai"""
        filename = Path(image_link).name
        image_path = os.path.join(self.image_folder, filename)
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {filename}")
            return np.zeros(512)  # Default zero features
        
        try:
            # Image load karke preprocess karo
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # CNN features extract karo - deep learning magic!
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            return np.zeros(512)
    
    def process_all_images(self, df):
        """Sabhi images ka features extract karo - batch processing"""
        print("🎨 Image features extract kar rahe hai...")
        
        features = []
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"📸 Processing image {idx+1}/{len(df)}")
            
            img_features = self.extract_image_features(row['image_link'])
            features.append(img_features)
        
        features = np.array(features)
        print(f"✅ Image features ready! Shape: {features.shape}")
        return features

class SmartEnsembleModel:
    """
    Ensemble model - multiple models ka combination!
    XGBoost + LightGBM = Double power! 🚀
    """
    
    def __init__(self, config):
        print("🎯 Ensemble Model initialize kar rahe hai...")
        self.config = config
        self.models = {'xgb': [], 'lgb': []}  # Model storage
        self.scaler = StandardScaler()  # Feature scaling
    
    def calculate_smape(self, y_true, y_pred):
        """SMAPE calculation - competition metric"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred)
        smape = np.mean(diff / denominator) * 100
        return smape
    
    def train_fold_models(self, X_train, y_train, X_val, y_val):
        """Ek fold ke liye models train karo"""
        fold_models = {}
        fold_scores = {}
        
        # XGBoost training - gradient boosting master!
        print("  🌟 XGBoost train kar rahe hai...")
        xgb_model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
        xgb_model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=30,
                     verbose=False)
        
        xgb_pred = xgb_model.predict(X_val)
        xgb_score = self.calculate_smape(y_val, xgb_pred)
        fold_models['xgb'] = xgb_model
        fold_scores['xgb'] = xgb_score
        
        # LightGBM training - Microsoft ka fast model!
        print("  ⚡ LightGBM train kar rahe hai...")
        lgb_model = lgb.LGBMRegressor(**self.config.LGB_PARAMS)
        lgb_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(30)])
        
        lgb_pred = lgb_model.predict(X_val)
        lgb_score = self.calculate_smape(y_val, lgb_pred)
        fold_models['lgb'] = lgb_model
        fold_scores['lgb'] = lgb_score
        
        print(f"  📊 Fold scores - XGB: {xgb_score:.2f}%, LGB: {lgb_score:.2f}%")
        return fold_models, fold_scores
    
    def train_ensemble(self, X, y):
        """Cross-validation se ensemble train karo - robust model banao!"""
        print("\n🏋️ Ensemble training start kar rahe hai...")
        
        # Feature scaling - important for ML models
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Fold cross-validation setup
        kfold = KFold(n_splits=self.config.N_FOLDS, shuffle=True, 
                     random_state=self.config.RANDOM_STATE)
        
        all_scores = {'xgb': [], 'lgb': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            print(f"\n📁 Fold {fold + 1}/{self.config.N_FOLDS}")
            
            # Data split
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train models for this fold
            fold_models, fold_scores = self.train_fold_models(X_train, y_train, X_val, y_val)
            
            # Store models and scores
            for model_name in ['xgb', 'lgb']:
                self.models[model_name].append(fold_models[model_name])
                all_scores[model_name].append(fold_scores[model_name])
        
        # Overall performance summary
        print("\n📈 Final Cross-Validation Results:")
        for model_name in ['xgb', 'lgb']:
            avg_score = np.mean(all_scores[model_name])
            std_score = np.std(all_scores[model_name])
            print(f"  {model_name.upper()}: {avg_score:.2f}% ± {std_score:.2f}%")
        
        # Ensemble average
        avg_ensemble = np.mean([np.mean(all_scores['xgb']), np.mean(all_scores['lgb'])])
        print(f"  🎯 Expected Ensemble SMAPE: ~{avg_ensemble:.1f}%")
    
    def predict(self, X):
        """Ensemble prediction - sabhi models ka average"""
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for model_name in ['xgb', 'lgb']:
            model_preds = []
            for model in self.models[model_name]:
                pred = model.predict(X_scaled)
                model_preds.append(pred)
            # Average across folds
            avg_pred = np.mean(model_preds, axis=0)
            predictions.append(avg_pred)
        
        # Ensemble average - equal weights
        final_pred = np.mean(predictions, axis=0)
        return final_pred

class SmartPricingPipeline:
    """
    Complete ML pipeline - end-to-end solution!
    Watson-style integrated approach 🎯
    """
    
    def __init__(self):
        print("🚀 Smart Pricing Pipeline initialize kar rahe hai...")
        self.config = SmartPricingConfig()
        
        # Feature engineers
        self.text_engineer = TextFeatureEngineer(self.config.TFIDF_MAX_FEATURES)
        self.image_engineer = None  # Initialize when needed
        
        # Ensemble model
        self.model = SmartEnsembleModel(self.config)
    
    def load_data(self):
        """Dataset load karo - pandas power!"""
        print("📂 Dataset load kar rahe hai...")
        
        train_path = os.path.join(self.config.DATASET_FOLDER, 'train.csv')
        test_path = os.path.join(self.config.DATASET_FOLDER, 'test.csv')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"✅ Data loaded - Train: {len(self.train_df)}, Test: {len(self.test_df)}")
        return self.train_df, self.test_df
    
    def create_features(self, use_images=False):
        """Feature engineering pipeline - multimodal approach!"""
        print("\n🛠️ Feature engineering start kar rahe hai...")
        
        # Text features - NLP magic
        train_text_features = self.text_engineer.process_text_features(self.train_df, is_training=True)
        test_text_features = self.text_engineer.process_text_features(self.test_df, is_training=False)
        
        if use_images:
            print("🖼️ Image features bhi include kar rahe hai...")
            # Initialize image engineer
            self.image_engineer = ImageFeatureEngineer(self.config.IMAGE_FOLDER)
            
            # Extract image features
            train_img_features = self.image_engineer.process_all_images(self.train_df)
            test_img_features = self.image_engineer.process_all_images(self.test_df)
            
            # Combine text + image features
            self.X_train = np.hstack([train_text_features, train_img_features])
            self.X_test = np.hstack([test_text_features, test_img_features])
            
        else:
            print("📝 Sirf text features use kar rahe hai - faster approach!")
            self.X_train = train_text_features
            self.X_test = test_text_features
        
        # Target variable
        self.y_train = self.train_df['price'].values
        
        print(f"✅ Features ready! Train: {self.X_train.shape}, Test: {self.X_test.shape}")
    
    def train_model(self):
        """Model training - ML magic time!"""
        print("\n🎓 Model training start kar rahe hai...")
        self.model.train_ensemble(self.X_train, self.y_train)
        print("✅ Model training complete!")
    
    def generate_predictions(self):
        """Final predictions generate karo - submission ready!"""
        print("\n🔮 Predictions generate kar rahe hai...")
        
        predictions = self.model.predict(self.X_test)
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'sample_id': self.test_df['sample_id'],
            'price': predictions
        })
        
        # Quality checks
        print("✅ Quality checks:")
        print(f"  Min price: ₹{submission['price'].min():.2f}")
        print(f"  Max price: ₹{submission['price'].max():.2f}")
        print(f"  Avg price: ₹{submission['price'].mean():.2f}")
        
        # Save submission
        output_path = os.path.join(self.config.DATASET_FOLDER, 'test_out.csv')
        submission.to_csv(output_path, index=False)
        
        print(f"🎯 Submission saved: {output_path}")
        return submission
    
    def run_complete_pipeline(self, use_images=True):
        """Complete pipeline execute karo - one-click solution!"""
        print("=" * 50)
        print("🚀 SMART PRICING CHALLENGE - PRAK SOLUTION")
        print("=" * 50)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Feature engineering  
            self.create_features(use_images=use_images)
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Generate predictions
            submission = self.generate_predictions()
            
            print("\n🎉 Pipeline complete! Ready for submission! 🏆")
            return submission
            
        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            raise

def main():
    """Main function - yahan se sab start hota hai!"""
    
    # Create pipeline instance
    pipeline = SmartPricingPipeline()
    
    # Run complete solution
    # use_images=False for faster execution
    # use_images=True for best accuracy 
    submission = pipeline.run_complete_pipeline(use_images=False)
    
    print("\n📊 Sample predictions:")
    print(submission.head(10))

if __name__ == "__main__":
    main()

# 💡 Pro Tips for Running:
# 1. Fast testing: python smart_pricing_hinglish.py (text-only)
# 2. Best accuracy: Set use_images=True in main()
# 3. GPU available: Automatic detection, image processing faster hogi
# 4. Memory issues: Reduce TFIDF_MAX_FEATURES in config
# 5. IBM Watson integration: API endpoints add kar sakte hai future mein!

print("🎯 Smart Pricing Solution by Prakhar - Ready to deploy!")