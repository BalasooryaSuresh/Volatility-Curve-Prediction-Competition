# Options Implied Volatility Prediction

A machine learning project for predicting options implied volatility using ensemble methods and advanced feature engineering techniques.

## Project Overview

This project implements sophisticated machine learning models to predict implied volatility (IV) for options contracts. The solution combines multiple regression algorithms with advanced feature engineering, volatility surface modeling, and ensemble techniques to achieve accurate predictions.

## Features

- **Advanced Feature Engineering**: Time-based features, rolling statistics, momentum indicators, and volatility surface characteristics
- **Ensemble Methods**: Combines LightGBM, XGBoost, Random Forest, Extra Trees, KNN, Ridge, and Elastic Net regressors
- **Volatility Surface Modeling**: Sophisticated interpolation and surface feature extraction
- **Missing Value Handling**: KNN imputation and spline-based interpolation for missing IV values
- **Robust Preprocessing**: StandardScaler, RobustScaler, and PCA for dimensionality reduction

## File Structure

```
├── README.md
├── .gitattributes                    # Git LFS configuration for large files
├── main.py                          # Simplified baseline model
├── claude_main.py                   # Advanced ensemble model with comprehensive features
├── train_data.parquet              # Training dataset (116MB)
├── test_data.parquet               # Test dataset (6.6MB)
├── sample_submission.csv           # Sample submission format
├── final_submission_optimized.csv  # Final submission file
└── data/
    ├── train-data.csv              # Training data in CSV format (167MB)
    └── test-data.csv               # Test data in CSV format (8.7MB)
```

## Data Description

### Input Features
- **X0-X41**: 42 anonymous numerical features
- **underlying**: Underlying asset price
- **timestamp**: Time information for temporal features

### Target Variables
- Multiple implied volatility columns for different option contracts
- Format: `{call/put}_iv_{strike_price}`
- Example: `call_iv_100`, `put_iv_95`

## Models

### Baseline Model (`main.py`)
- **Algorithm**: HistGradientBoostingRegressor with MultiOutput wrapper
- **Features**: Basic feature engineering with PCA
- **Preprocessing**: StandardScaler + SimpleImputer
- **Performance**: Quick baseline solution

### Advanced Model (`claude_main.py`)
- **Algorithms**: Ensemble of 8 different regressors
- **Feature Engineering**: 
  - Rolling statistics (3, 5, 10, 20 periods)
  - Momentum and volatility indicators
  - Volatility surface features
  - Cross-features with X variables
  - Time-based cyclical features
- **Preprocessing**: 
  - KNN imputation for missing values
  - Spline interpolation for missing IVs
  - RobustScaler for feature scaling
- **Ensemble Weights**: Optimized weighting favoring gradient boosting methods

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd options-iv-prediction

# Install required packages
pip install pandas numpy scikit-learn lightgbm xgboost scipy

# Install Git LFS for large files
git lfs install
git lfs pull
```

## Usage

### Quick Start (Baseline Model)
```bash
python main.py
```

### Advanced Model
```bash
python claude_main.py
```

Both scripts will:
1. Load and preprocess the data
2. Train the models
3. Generate predictions
4. Save submission files

## Model Architecture

### Advanced Ensemble Components

1. **LightGBM Regressors** (2 variants)
   - Primary models with different hyperparameters
   - Weight: 45% of ensemble

2. **XGBoost Regressor**
   - Gradient boosting with regularization
   - Weight: 20% of ensemble

3. **Tree-based Models**
   - Random Forest and Extra Trees
   - Weight: 20% of ensemble

4. **Distance-based and Linear Models**
   - KNN, Ridge Regression, Elastic Net
   - Weight: 15% of ensemble

### Feature Engineering Pipeline

1. **Temporal Features**
   - Hour, minute, second extraction
   - Cyclical encoding (sin/cos transformations)
   - Time-based rolling statistics

2. **Underlying Asset Features**
   - Log transformation and normalization
   - Rolling means and standard deviations
   - Momentum and acceleration indicators

3. **Volatility Surface Features**
   - Call/Put IV statistics (mean, std, min, max, range)
   - Put-Call parity relationships
   - Strike-based interpolation

4. **Cross-feature Engineering**
   - X-variable statistics (sum, mean, std, skew, kurtosis)
   - Correlation with underlying asset

## Performance Optimization

- **Memory Efficiency**: Uses parquet format for faster I/O
- **Parallel Processing**: Multi-core training where supported
- **Missing Value Strategy**: Advanced interpolation preserves volatility surface structure
- **Bounds Enforcement**: IV predictions clipped to reasonable ranges (0.01-2.0)

## Configuration

Key hyperparameters can be adjusted in the model creation:

```python
# LightGBM parameters
n_estimators=500
learning_rate=0.01
max_depth=6
reg_alpha=0.1
reg_lambda=0.1

# Ensemble weights
weights = [0.25, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05]
```

## Output

The models generate CSV files with predicted implied volatilities:
- `final_optimized_submission.csv` (from main.py)
- `advanced_optimized_submission.csv` (from claude_main.py)

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
xgboost>=1.5.0
scipy>=1.7.0
```

## Notes

- Large data files are managed with Git LFS
- Models include extensive error handling for robustness
- IV predictions are bounded to prevent unrealistic values
- The advanced model prioritizes gradient boosting methods in the ensemble

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.
