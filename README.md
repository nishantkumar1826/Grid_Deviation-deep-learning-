# ⭐ GRID DEVIATION – DEEP LEARNING MODEL
### 🔮 *Predicting Power Grid Deviation using XGBoost + ML Pipelines*



---

## 🧩 3D PROJECT OVERVIEW (Enhanced)

This project builds a **high-accuracy Machine Learning model** to predict **Grid Deviation (MW)** using engineered features and an optimized XGBoost training pipeline.

It includes:

- Data ingestion  
- Preprocessing & feature engineering  
- Model training with XGBoost  
- SMAPE/Metrics evaluation  
- Feature importance  
- Test predictions + submission file  

**Validation Metrics:**

| Metric | Score |
|--------|--------|
| **MAE** | 1.3020 |
| **RMSE** | 16.0705 |
| **R²** | 0.9497 |
| **SMAPE Accuracy** | (computed via notebook) |

---

# 🔷 3D STACK DIAGRAM — Complete Workflow

```
               ┌────────────────────────────┐
               │      DATA INGESTION        │
               │  (CSV / Kaggle / Downloads)│
               └────────────┬───────────────┘
                            │
                ┌───────────▼───────────┐
                │   PREPROCESSING &     │
                │  FEATURE ENGINEERING  │
                │ (dates → epoch, lags, │
                │  rolling stats, enc)  │
                └───────┬─────────┬──────┘
                        │         │
      ┌─────────────────▼─┐     ┌─▼─────────────────┐
      │  MODEL TRAINING    │     │  VALIDATION &     │
      │  (XGBoost / Opt)   │     │  EVALUATION (SMAPE│
      │  hyperparams)      │     │   MAE RMSE R²)    │
      └─────┬──────────────┘     └────┬──────────────┘
            │                           │
            └───────────┬───────────────┘
                        ▼
            ┌────────────────────────────┐
            │  DEPLOY / SUBMISSION CSV   │
            │  (submission.csv / model)  │
            └────────────────────────────┘
```

---

# 🔶 3D ISOMETRIC BLOCK — Architecture View

```
           .──────────────.
          /  DATA LAYER    \
         /  (kaggle/input)  \
        /____________________\
       /  .──────────────.    \
      /  /  PREPROCESSING\    \
     /  /  & ENCODING    \    \
    /  /__________________\    \
   /   .──────────────.   \    \
  /   /  MODEL LAYER    \   \   \
 /   /   XGBoost / LGB   \   \   \
/___/_____________________\___\___\
\   \   .──────────.   .────────.  /
 \   \ / EVAL &    \ / FEATURE  \ /
  \   X   METRICS   X  IMPORTANCE X
   \ / \ (SMAPE etc)/ \  (plots) / 
    '----------------' '--------'
```

---

# 🔺 3D FLOWCHART — Data → Model → Insights

```
 [raw CSV] --> [cleaning] --> [lag features] --> [train/test split]
     |              |                 |                  |
     v              v                 v                  v
 [missing fill] [date→epoch] [rolling mean/std] --> [XGBoost training]
                                                   |
                                                   v
                                         [Feature importance chart]
                                                   |
                                                   v
                                             [submission.csv]
```

---

# 📂 PROJECT STRUCTURE

```
Grid_Deviation/
│── kaggle/                         # Dataset (ignored)
│── notebooks/
│    └── kindle-kids-grid-deviation.ipynb
│── src/
│    ├── train.py
│    └── predict.py
│── models/
│    └── xgb_model_optimized.joblib
│── outputs/
│    ├── feature_importance_top.png
│    └── training_progress.png
│── submission.csv
│── requirements.txt
│── README.md
│── .gitignore
```

---

# 🚀 INSTALLATION & QUICK RUN

```bash
git clone https://github.com/nishantkumar1826/Grid_Deviation-deep-learning-.git
cd Grid_Deviation-deep-learning-
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# OR
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

Open and run the notebook:

```
notebooks/kindle-kids-grid-deviation.ipynb
```

OR if you convert to python scripts:

```
python src/train.py
```

---

# 📊 MODEL EVALUATION (VALIDATION SET)

```
MAE   : 1.3020
RMSE  : 16.0705
R²    : 0.9497          (~94.97% of variance explained)
```

---

# 🔄 SMAPE (Recommended Accuracy Formula)

```
SMAPE = (100% / n) * Σ( 2 * |pred - true| / (|true| + |pred| + eps) )
Accuracy ≈ 100 - SMAPE
```

SMAPE avoids exploding errors when true values ≈ 0.

---

# 🧬 FEATURE IMPORTANCE

Feature importance (Gain-based) is saved as:

```
outputs/feature_importance_top.png
```

Use this for interpretability and model debugging.

---

# 💾 SUBMISSION FILE FORMAT

Created automatically:

```
index,prediction
0,-26.17645
1,-23.57566
2,-55.60978
...
```

Saved as **submission.csv**

---

# 🛠 FUTURE IMPROVEMENTS

- Hyperparameter tuning (Optuna)
- LSTM/GRU time-series deep learning version
- AutoML experimentation (H2O / PyCaret)
- Real-time grid deviation dashboard (Plotly/Streamlit)
- REST API for prediction (FastAPI)

---

# ✨ AUTHOR

**Nishant Kumar**  
GitHub: https://github.com/nishantkumar1826  

---

# ❤️ NEED MORE?
I can provide:

- custom **animated project banner**,  
- vector art,  
- better diagrams,  
- a full **requirements.txt**,  
- a professional `.gitignore`.

Just tell me!
### 🔗 Connect With Me  
<a href="https://www.linkedin.com/in/nishant-kumar-92b07b381/" target="_blank">LinkedIn Profile</a>


