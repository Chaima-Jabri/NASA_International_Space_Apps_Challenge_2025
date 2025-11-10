<div align="center">

# 🌟 Exoplanet KeplerAI

### AI-Powered Exoplanet Detection System


</div>

---

## 📖 About

**ExoKeplerAI** is an advanced AI system designed to automatically analyze NASA Kepler satellite data and identify exoplanets with high accuracy. The system combines three state-of-the-art machine learning algorithms (LightGBM, CatBoost, and XGBoost) using ensemble learning to achieve optimal classification performance.

### 🎯 Key Objectives

- **Automate** the detection of exoplanets from Kepler mission data
- **Reduce** manual analysis time for astronomers
- **Accelerate** the discovery of new exoplanets
- **Provide** an educational tool for the astronomy community

---

## ✨ Features

### 🤖 Machine Learning
- **Ensemble Learning**: Combines LightGBM, CatBoost, and XGBoost
- **High Accuracy**: 92.79% classification accuracy
- **Intelligent Preprocessing**: Automated feature engineering and normalization
- **Class Balancing**: Handles imbalanced datasets effectively

### 🌐 Web Interface
- **Interactive Dashboard**: Built with Gradio
- **Real-time Predictions**: Instant classification results
- **Batch Processing**: Analyze multiple observations via CSV upload
- **Visualizations**: Confidence gauges, probability charts, and performance metrics

### 📊 Data Analysis
- **9,564 Observations**: Comprehensive Kepler dataset
- **49 Features**: Including orbital period, transit duration, planetary radius, etc.
- **3 Classes**: CONFIRMED, CANDIDATE, FALSE POSITIVE
- **Feature Importance**: Detailed analysis of key predictive features

---

## 🚀 Quick Start

### Installation (3 Steps)

```bash
# 1. Clone the repository
git clone link
cd ExoKeplerAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (5-10 minutes)
python train_model.py
```

### Launch Web Interface

```bash
python app.py
```

Then open: **http://localhost:7860**

---

## 💻 Usage

### Web Interface

#### Simple Prediction
1. Navigate to the **"🔍 Simple Prediction"** tab
2. Click **"Load Example"** to populate with sample data
3. Click **"Predict"** to get results
4. View prediction, confidence level, and probability distribution

#### Batch Processing (CSV)
1. Navigate to the **"📊 Batch Prediction (CSV)"** tab
2. Upload a CSV file with the required 17 columns
3. Click **"Analyze CSV"**
4. View statistics and detailed results


## 📊 Performance

### Model Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.79% |
| **Precision** | 92.92% |
| **Recall** | 92.79% |
| **F1-Score** | 92.83% |

### Top 5 Most Important Features

1. **confidence_score** (468.95) - Engineered feature
2. **koi_duration** (437.67) - Transit duration
3. **koi_kepmag** (409.44) - Kepler magnitude
4. **koi_model_snr** (361.31) - Signal-to-noise ratio
5. **koi_impact** (322.03) - Impact parameter

### Class Distribution

- **FALSE POSITIVE**: 4,839 (50.6%)
- **CONFIRMED**: 2,746 (28.7%)
- **CANDIDATE**: 1,979 (20.7%)

---

## 🔬 Methodology

### 1. Data Preprocessing

- **Missing Value Imputation**: Median strategy for numerical features
- **Feature Engineering**: 5 new features created from existing data
  - `prad_srad_ratio`: Planetary/stellar radius ratio
  - `density_proxy`: Approximate density indicator
  - `confidence_score`: Composite confidence metric
  - `total_fp_flags`: Sum of false positive flags
  - `in_habitable_zone`: Binary habitable zone indicator
- **Normalization**: StandardScaler for feature scaling
- **Class Balancing**: Weighted classes to handle imbalance

### 2. Model Architecture

**Ensemble Learning with Weighted Voting:**

| Model | Weight | Key Strengths |
|-------|--------|---------------|
| **LightGBM** | 33% | Fast training, efficient memory usage |
| **CatBoost** | 34% | Handles categorical features, robust |
| **XGBoost** | 33% | High accuracy, regularization |

**Hyperparameters:**
- 500 estimators per model
- Learning rate: 0.05
- Max depth: 8
- Early stopping: 50 rounds

### 3. Evaluation

- **Train/Test Split**: 80/20
- **Stratified Sampling**: Maintains class distribution
- **Cross-Validation**: Validation set for early stopping
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 🛠️ Technologies

### Machine Learning
- **scikit-learn** - Preprocessing and metrics
- **LightGBM** - Gradient boosting framework
- **CatBoost** - Categorical boosting
- **XGBoost** - Extreme gradient boosting
- **imbalanced-learn** - Class imbalance handling

### Web Interface
- **Gradio** - Interactive web UI
- **Plotly** - Interactive visualizations
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **joblib** - Model serialization

---

## 🎓 Developed For

**NASA Space Apps Challenge 2025**

This project was created to address the challenge of automatically identifying exoplanets from NASA Kepler mission data, reducing the need for manual analysis and accelerating the pace of exoplanet discovery.


<div align="center">

**Made with ❤️ for exoplanet discovery**

</div>
