# Quick Start Guide - Smart Product Pricing Challenge

## 🚀 Getting Started in 5 Minutes

### Option 1: Fast Baseline (Recommended for Testing)

**Best for:** Quick iteration, testing your setup, initial submission

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm

# 2. Run the simplified solution (text-only)
python simplified_pricing_solution.py
```

**Expected time:** 10-15 minutes  
**Expected SMAPE:** ~15-18% on validation

---

### Option 2: Full Solution (Best Performance)

**Best for:** Final submission, maximum accuracy

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Run full pipeline with images
python complete_pricing_solution.py --mode full

# OR run without images (faster)
python complete_pricing_solution.py --mode text_only
```

**Expected time:** 2-3 hours (with images), 30-45 mins (text only)  
**Expected SMAPE:** ~13-15% on validation (with images)

---

## 📋 Step-by-Step Instructions

### Step 1: Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Data Structure

Ensure your directory structure looks like this:

```
project/
├── dataset/
│   ├── train.csv          # 75k training samples
│   ├── test.csv           # 75k test samples
│   ├── sample_test.csv
│   └── sample_test_out.csv
├── src/
│   └── utils.py
├── complete_pricing_solution.py
├── simplified_pricing_solution.py
├── requirements.txt
└── images/                # Will be created automatically
```

### Step 3: Choose Your Approach

#### A. Quick Baseline (Recommended First)

```python
# This runs in ~15 minutes and gives you a working submission
python simplified_pricing_solution.py
```

**Output:** `dataset/test_out.csv`

#### B. Full Pipeline with Cross-Validation

```bash
# Full pipeline with images (best accuracy)
python complete_pricing_solution.py --mode full

# Text-only pipeline (faster, still good)
python complete_pricing_solution.py --mode text_only

# Quick baseline mode
python complete_pricing_solution.py --mode baseline
```

### Step 4: Check Your Output

```python
import pandas as pd

# Load your predictions
output = pd.read_csv('dataset/test_out.csv')

# Verify format
print(output.head())
print(f"Shape: {output.shape}")  # Should be (75000, 2)
print(f"Columns: {list(output.columns)}")  # Should be ['sample_id', 'price']
print(f"Min price: {output['price'].min()}")  # Should be > 0
print(f"Max price: {output['price'].max()}")
```

---

## 🎯 Performance Expectations

| Approach | Time | SMAPE | Features Used |
|----------|------|-------|---------------|
| Simplified Baseline | 15 min | ~16-18% | Text only, 2 models |
| Text-Only Full | 45 min | ~14-16% | Text + embeddings, 3 models |
| Full with Images | 2-3 hrs | ~13-15% | Text + images, 3 models |

---

## 🔧 Troubleshooting

### Issue: Image Download Fails

**Problem:** Throttling or network issues

**Solution:**
```python
# Option 1: Run without images
python complete_pricing_solution.py --no_images

# Option 2: Download images separately with retry
from src.utils import download_images
import pandas as pd

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# This will skip already downloaded images
download_images(train['image_link'].tolist(), 'images/')
download_images(test['image_link'].tolist(), 'images/')
```

### Issue: Out of Memory

**Problem:** Not enough RAM for full dataset

**Solutions:**
```bash
# 1. Use simplified version (lower memory)
python simplified_pricing_solution.py

# 2. Reduce feature dimensions in config
# Edit Config class in complete_pricing_solution.py
TFIDF_MAX_FEATURES = 1000  # Instead of 3000
N_FOLDS = 3  # Instead of 5

# 3. Process images in batches
# Already handled in the code
```

### Issue: CUDA/GPU Not Available

**Problem:** PyTorch can't find GPU

**Solution:**
```python
# The code automatically falls back to CPU
# To force CPU:
import torch
device = torch.device('cpu')

# Or install CPU-only PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Missing Dependencies

**Problem:** Import errors

**Solution:**
```bash
# Install specific missing packages
pip install xgboost lightgbm catboost
pip install sentence-transformers
pip install timm

# Or install everything
pip install -r requirements.txt
```

---

## 📊 Improving Your Score

### Quick Wins (Easy)

1. **Tune Ensemble Weights**
   ```python
   # In complete_pricing_solution.py, line ~350
   weights = [0.35, 0.35, 0.30]  # Experiment with these
   weights = [0.4, 0.3, 0.3]     # Try different combinations
   ```

2. **Adjust TF-IDF Features**
   ```python
   # In Config class
   TFIDF_MAX_FEATURES = 5000  # Increase from 3000
   ```

3. **Add More Folds**
   ```python
   # In Config class
   N_FOLDS = 7  # Increase from 5 (more stable, takes longer)
   ```

### Advanced Improvements (Harder)

1. **Better Text Features**
   - Use larger sentence transformers (e.g., 'all-mpnet-base-v2')
   - Extract brand names with regex patterns
   - Create category-specific features

2. **Better Image Features**
   - Use EfficientNet-B4 instead of ResNet-18
   - Extract CLIP features for vision-language alignment
   - OCR text from product images

3. **Advanced Ensembling**
   - Stack models (train meta-model on predictions)
   - Use optuna for hyperparameter optimization
   - Train separate models for different price ranges

4. **Data Augmentation** (Allowed)
   - Text augmentation: synonym replacement, back-translation
   - Feature engineering: interaction features, polynomial features

---

## 🎓 Best Practices

### 1. Start Simple, Then Iterate

```bash
# Day 1: Get baseline working
python simplified_pricing_solution.py

# Day 2: Add more features
python complete_pricing_solution.py --mode text_only

# Day 3: Add images
python complete_pricing_solution.py --mode full

# Day 4-5: Tune and optimize
```

### 2. Always Validate Locally

```python
# Check your validation SMAPE before submitting
# It should correlate with leaderboard score
```

### 3. Version Control Your Experiments

```bash
# Save different versions
mv dataset/test_out.csv submissions/baseline_v1.csv
mv dataset/test_out.csv submissions/with_images_v2.csv
```

### 4. Document What Works

Keep a log:
```
Experiment 1: Baseline - 17.2% SMAPE
Experiment 2: + embeddings - 15.8% SMAPE  
Experiment 3: + images - 14.3% SMAPE ✓ Best so far
Experiment 4: + tuned weights - 14.1% SMAPE ✓ New best
```

---

## 📝 Submission Checklist

Before submitting, verify:

- [ ] Output file is named `test_out.csv`
- [ ] File has exactly 75,000 rows (same as test.csv)
- [ ] Columns are: `sample_id`, `price` (in that order)
- [ ] No missing values in predictions
- [ ] All prices are positive (> 0)
- [ ] Sample IDs match test.csv exactly
- [ ] File format matches sample_test_out.csv

**Verification script:**
```python
import pandas as pd

test = pd.read_csv('dataset/test.csv')
output = pd.read_csv('dataset/test_out.csv')
sample_out = pd.read_csv('dataset/sample_test_out.csv')

# Check 1: Row count
assert len(output) == len(test), f"Wrong row count: {len(output)} vs {len(test)}"

# Check 2: Columns
assert list(output.columns) == ['sample_id', 'price'], f"Wrong columns: {output.columns}"

# Check 3: Sample IDs match
assert set(output['sample_id']) == set(test['sample_id']), "Sample IDs don't match!"

# Check 4: No nulls
assert not output.isnull().any().any(), "Found null values!"

# Check 5: Positive prices
assert (output['price'] > 0).all(), "Found non-positive prices!"

# Check 6: Format matches
assert output.dtypes['sample_id'] == sample_out.dtypes['sample_id']
assert output.dtypes['price'] == sample_out.dtypes['price']

print("✓ All checks passed! Ready to submit.")
print(f"\nPrice stats:")
print(output['price'].describe())
```

---

## 🚀 Advanced Usage

### Custom Predictor Function

If you want to integrate with sample_code.py:

```python
from simplified_pricing_solution import SimplePricingModel
import pandas as pd

# Train model once
train_df = pd.read_csv('dataset/train.csv')
model = SimplePricingModel()
model.train(train_df)

# Use in predictor function
def predictor(sample_id, catalog_content, image_link):
    # Create single-row dataframe
    df = pd.DataFrame({
        'catalog_content': [catalog_content],
        'image_link': [image_link]
    })
    
    # Get prediction
    price = model.predict(df)[0]
    return round(price, 2)
```

### Parallel Processing

For faster image processing:

```python
# Already implemented in utils.py with multiprocessing
# Adjust number of workers:
with multiprocessing.Pool(50) as pool:  # Reduce if unstable
    # ... download images
```

### Batch Prediction

For memory efficiency:

```python
# Process test set in batches
batch_size = 5000
all_predictions = []

for i in range(0, len(test_df), batch_size):
    batch = test_df.iloc[i:i+batch_size]
    batch_features = prepare_features(batch)
    batch_preds = model.predict(batch_features)
    all_predictions.extend(batch_preds)
```

---

## 📚 Additional Resources

- **Challenge Discussion:** [Forum Link]
- **Gradient Boosting Tutorials:** 
  - XGBoost: https://xgboost.readthedocs.io/
  - LightGBM: https://lightgbm.readthedocs.io/
- **Computer Vision:**
  - PyTorch Vision: https://pytorch.org/vision/
  - Timm Models: https://github.com/rwightman/pytorch-image-models
- **NLP:**
  - Sentence Transformers: https://www.sbert.net/

---

## 💡 Tips from Experience

1. **Don't over-optimize on validation set** - Use cross-validation
2. **Image features help most for mid-high prices** - Low-price products depend more on text
3. **Ensemble different model types** - Not just different hyperparameters
4. **Handle edge cases** - Missing images, very long text, special characters
5. **Monitor training time** - Balance accuracy vs. computation cost

---

## 🎉 Good Luck!

You're now ready to build a competitive solution. Start with the simplified baseline, verify it works, then gradually add complexity.

**Remember:** The best solution is one that:
- Works reliably (no errors)
- Validates properly (good CV score)
- Generalizes well (similar CV and LB scores)
- Is documented clearly

Happy modeling! 🚀