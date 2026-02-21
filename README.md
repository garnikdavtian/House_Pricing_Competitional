# House Prices: Advanced Regression Techniques TOP 5% RESULT ON KAGGLE

Predicting residential property sale prices in Ames, Iowa using a 7-model stacking ensemble with 30+ engineered features.

**Competition:** [Kaggle -- House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
**Metric:** RMSE on log-transformed SalePrice (RMSLE)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Strategy](#strategy)
- [Models Used](#models-used)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Dataset](#dataset)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project tackles the classic Kaggle House Prices regression competition. The goal is to predict the final sale price of homes in Ames, Iowa, given 79 explanatory variables describing nearly every aspect of residential homes.

The solution achieves a competitive RMSLE score through a combination of thorough exploratory data analysis, aggressive feature engineering (30+ new features), and a multi-model stacking ensemble with weighted blending.

---

## Project Structure

```
House_Pricing_competition/
|
|-- data/
|   |-- train.csv                # Training dataset (1,460 samples, 80 features)
|   |-- test.csv                 # Test dataset (1,459 samples, 79 features)
|   |-- sample_submission.csv    # Sample submission format
|   |-- data_description.txt     # Full description of all features
|
|-- solution.ipynb               # Main notebook with the full pipeline
|-- requirements.txt             # Python dependencies
|-- LICENSE                      # MIT License
|-- .gitignore                   # Git ignore rules
|-- README.md                    # This file
```

---

## Strategy

The solution follows a structured 5-phase approach:

| Phase | Technique | Purpose |
|-------|-----------|---------|
| **1. EDA** | Distribution analysis, correlation study, outlier detection | Understand data structure |
| **2. Preprocessing** | Smart imputation, ordinal encoding, skewness correction | Clean and prepare data |
| **3. Feature Engineering** | 30+ engineered features: polynomials, interactions, target encoding | Maximize predictive signal |
| **4. Modeling** | 7-model stacking ensemble (Lasso, ElasticNet, Ridge, KRR, GBR, XGB, LGB) | Diverse predictions |
| **5. Blending** | Weighted average of stacked predictions + individual boosters | Robust final prediction |

---

## Models Used

The ensemble consists of 7 base models combined through scikit-learn's `StackingRegressor` architecture:

### Base Learners (Stacking)
- **Lasso Regression** -- L1 regularized linear model for feature selection
- **ElasticNet** -- Combines L1 and L2 regularization
- **Ridge Regression** -- L2 regularized linear model
- **Kernel Ridge Regression (KRR)** -- Non-linear extension of Ridge
- **Gradient Boosting Regressor (GBR)** -- Scikit-learn's gradient boosting

### Standalone Boosters (Blending)
- **XGBoost** -- Extreme gradient boosting
- **LightGBM** -- Light gradient boosting machine

### Blending Weights
The final prediction is a weighted blend:
- **Stacking Ensemble:** 70%
- **XGBoost:** 15%
- **LightGBM:** 15%

All models operate in log-space (`log1p`/`expm1` transform on SalePrice) for variance stabilization.

---

## Feature Engineering

Over 30 features are engineered to capture domain-specific signals:

- **Area composites:** Total square footage, total porch area, total bathrooms
- **Quality interactions:** Quality x area products, quality scores
- **Temporal features:** House age, years since remodel, age at sale
- **Polynomial features:** Squared and cubed terms for top correlated features
- **Ordinal encoding:** Quality/condition grades mapped to numeric scales
- **Skewness correction:** Box-Cox transformation on heavily skewed numeric features
- **Target encoding:** Neighborhood-level sale price statistics

---

## Results

| Metric | Value |
|--------|-------|
| Training set | 1,460 samples |
| Test set | 1,459 samples |
| Engineered features | 30+ |
| Models in ensemble | 7 |
| Cross-validation | 10-fold |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/garnikdavtian/House_Pricing_Competitional.git
cd House_Pricing_Competitional
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook solution.ipynb
```

Run all cells sequentially. The notebook will:
1. Load and explore the Ames Housing dataset
2. Preprocess and engineer features
3. Train the 7-model stacking ensemble
4. Generate blended predictions on the test set
5. Save `submission.csv` for Kaggle upload

---

## Dataset

The dataset is provided by the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

- **train.csv** -- 1,460 residential home sales with 79 features + SalePrice
- **test.csv** -- 1,459 homes to predict
- **data_description.txt** -- Detailed description of every feature

Key features include:
- `OverallQual` -- Overall material and finish quality (1-10)
- `GrLivArea` -- Above grade living area (sq ft)
- `TotalBsmtSF` -- Total basement area (sq ft)
- `GarageCars` -- Garage capacity
- `YearBuilt` -- Original construction date
- `Neighborhood` -- Physical location within Ames

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for hosting the competition and dataset
- Dean De Cock for compiling the Ames Housing dataset
- The scikit-learn, XGBoost, and LightGBM open-source communities
