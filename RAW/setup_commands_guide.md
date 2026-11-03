# 🚀 Smart Pricing Challenge - Project Setup Guide (Kali Linux)

## Step-by-Step Terminal Commands

### Phase 1: Project Directory Setup
```bash
# Main project folder banao
mkdir ~/smart-pricing-challenge
cd ~/smart-pricing-challenge

# Folder structure create karo - organized approach
mkdir -p dataset images src submissions docs

# Python virtual environment setup - clean dependencies
python3 -m venv pricing_env
source pricing_env/bin/activate

# Git repository initialize karo (Watson style project management)
git init
git remote add origin https://github.com/prak05/smart-pricing-watson-ai.git
```

### Phase 2: Required Files Creation
```bash
# Main solution file create karo
touch smart_pricing_hinglish.py

# Configuration aur utility files
touch src/__init__.py
touch src/utils.py
touch src/config.py

# Requirements file banao
touch requirements.txt

# Documentation files - professional approach
touch README.md
touch docs/solution_approach.md
touch docs/watson_integration.md

# Git files
touch .gitignore
```

### Phase 3: Dependencies Installation
```bash
# Basic requirements file mein add karo
cat > requirements.txt << 'EOF'
# Core ML Libraries - Data Science Stack
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0

# Gradient Boosting Models - Competition Winners
xgboost>=1.7.0
lightgbm>=3.3.0

# Deep Learning & Image Processing - CNN Power
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0

# NLP Libraries - Watson-style Text Processing
sentence-transformers>=2.2.0
transformers>=4.30.0

# Utilities & Visualization
tqdm>=4.65.0
requests>=2.28.0
matplotlib>=3.7.0
seaborn>=0.12.0

# IBM Watson SDK (Optional - for future integration)
ibm-watson>=7.0.0
ibm-cloud-sdk-core>=3.16.0
EOF

# Dependencies install karo - ye time lagega
pip install -r requirements.txt

# Additional IBM Watson libraries (optional)
pip install ibm-watsonx-ai>=0.1.0
```

### Phase 4: Dataset Setup
```bash
# Dataset files ko proper location mein move karo
# (Assume karte hai ki train.csv aur test.csv available hai)
cp ~/Downloads/train.csv dataset/
cp ~/Downloads/test.csv dataset/

# Sample files bhi copy karo
cp ~/Downloads/sample_test.csv dataset/
cp ~/Downloads/sample_test_out.csv dataset/

# Verify dataset structure
ls -la dataset/
echo "Dataset files:"
head -2 dataset/train.csv
```

### Phase 5: Utility Functions Create Karo
```bash
cat > src/utils.py << 'EOF'
#!/usr/bin/env python3
"""
Utility functions for Smart Pricing Challenge
Watson-style helper functions
"""

import os
import requests
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

def download_image(args):
    """Download single image with error handling"""
    url, folder = args
    try:
        filename = Path(url).name
        filepath = os.path.join(folder, filename)
        
        if os.path.exists(filepath):
            return True  # Already exists
            
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except:
        return False

def download_images(image_urls, folder, max_workers=20):
    """Parallel image downloading - Watson efficiency!"""
    os.makedirs(folder, exist_ok=True)
    
    args = [(url, folder) for url in image_urls]
    
    with mp.Pool(max_workers) as pool:
        results = list(tqdm(
            pool.imap(download_image, args),
            total=len(args),
            desc="Downloading images"
        ))
    
    success_count = sum(results)
    print(f"Downloaded {success_count}/{len(image_urls)} images")
    return success_count
EOF
```

### Phase 6: Main Solution File Setup
```bash
# Copy karo hamara Hinglish solution
cp smart_pricing_hinglish.py ./

# Executable banao
chmod +x smart_pricing_hinglish.py

# Test run karo (dry run)
python smart_pricing_hinglish.py --help 2>/dev/null || echo "Ready to run!"
```

### Phase 7: Git Setup & Version Control
```bash
# .gitignore file create karo
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
pricing_env/

# Data files (large files)
dataset/*.csv
images/*.jpg
images/*.png
*.pkl
*.h5

# Output files  
submissions/
test_out.csv

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db
EOF

# Git commits
git add .
git commit -m "Initial commit: Smart Pricing Challenge setup"
git branch -M main
git push -u origin main
```

## Execution Commands

### Quick Test Run (Text-only, Fast)
```bash
# Virtual environment activate karo
source pricing_env/bin/activate

# Basic test run - 15-20 minutes
python smart_pricing_hinglish.py

# Output check karo
head dataset/test_out.csv
wc -l dataset/test_out.csv  # Should show 75001 lines (header + 75000 predictions)
```

### Full Pipeline Run (With Images)
```bash
# Download images first (optional, for best accuracy)
python -c "
import pandas as pd
from src.utils import download_images

# Load datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Download images
download_images(train_df['image_link'].tolist(), 'images/')
download_images(test_df['image_link'].tolist(), 'images/')
"

# Run full pipeline - 1-2 hours
python smart_pricing_hinglish.py --use_images
```

### Performance Monitoring
```bash
# System resources monitor karo
htop &  # System monitoring

# GPU usage (if available)
watch nvidia-smi

# Memory usage track karo
watch "free -h && echo '---' && ps aux | grep python"
```

## IBM Watson Integration Commands

### Watson API Setup
```bash
# IBM Cloud CLI install karo
curl -fsSL https://clis.cloud.ibm.com/install/linux | sh

# Login to IBM Cloud
ibmcloud login

# Watson services list karo
ibmcloud resource service-instances

# API key create karo
ibmcloud iam api-key-create smart-pricing-key -d "Smart Pricing Challenge API Key"
```

### Watson X.ai Integration (Future Enhancement)
```bash
# Watson X.ai Python SDK
pip install ibm-watsonx-ai

# Environment variables set karo
export WATSON_API_KEY="your_api_key_here"
export WATSON_URL="https://us-south.ml.cloud.ibm.com"
export WATSON_PROJECT_ID="your_project_id"

# Test Watson connection
python -c "
from ibm_watsonx_ai import APIClient
api_client = APIClient(credentials={'apikey': os.getenv('WATSON_API_KEY')})
print('Watson X.ai connected successfully!')
"
```

### Watson Services Integration Examples
```bash
# Create Watson integration file
cat > src/watson_integration.py << 'EOF'
"""
IBM Watson integration for Smart Pricing Challenge
Future enhancement module
"""

import os
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

class WatsonPricingEnhancer:
    def __init__(self):
        self.api_key = os.getenv('WATSON_API_KEY')
        self.url = os.getenv('WATSON_URL')
    
    def enhance_text_features(self, text):
        """Use Watson NLP for advanced text processing"""
        # Implementation with Watson NLP services
        pass
    
    def watson_sentiment_analysis(self, product_descriptions):
        """Watson sentiment analysis for pricing insights"""
        # Implementation with Watson Tone Analyzer
        pass
EOF
```

## Directory Structure Verification
```bash
# Final project structure check karo
tree -a . -I 'pricing_env|__pycache__|*.pyc'

# Should look like this:
# .
# ├── .git/
# ├── .gitignore
# ├── README.md
# ├── requirements.txt
# ├── smart_pricing_hinglish.py
# ├── dataset/
# │   ├── train.csv
# │   ├── test.csv
# │   └── sample_test_out.csv
# ├── images/
# ├── src/
# │   ├── __init__.py
# │   ├── utils.py
# │   └── watson_integration.py
# ├── docs/
# │   └── solution_approach.md
# └── submissions/
```

## Pro Tips for Kali Linux Users 💪

### Performance Optimization
```bash
# CPU cores check karo
nproc

# Memory check karo  
free -h

# GPU check karo (if available)
lspci | grep -i nvidia
```

### Troubleshooting Commands
```bash
# Python path issues fix karo
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Permission issues fix karo
chmod +x *.py

# Dependency conflicts resolve karo
pip install --force-reinstall --no-deps package_name
```

### System Resource Management
```bash
# Swap memory increase karo (if needed)
sudo swapon -s
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Cleanup after completion
sudo swapoff /swapfile
sudo rm /swapfile
```

Bhai, ye complete setup guide hai! Follow karo step-by-step aur tumhara Watson-style pricing solution ready ho jayega! 🚀