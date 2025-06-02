import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedIVPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.iv_columns = []
        self.feature_columns = []
        self.strike_info = {}
        
    def extract_strike_info(self, iv_columns):
        """Extract strike information from IV column names"""
        strike_info = {}
        for col in iv_columns:
            parts = col.split('_')
            option_type = parts[0]  # 'call' or 'put'
            strike = int(parts[2])  # strike price
            strike_info[col] = {'type': option_type, 'strike': strike}
        return strike_info
    
    def advanced_feature_engineering(self, df, is_train=True):
        """Create sophisticated features for options pricing"""
        df_feat = df.copy()
        
        # Basic underlying features
        if 'underlying' in df_feat.columns:
            df_feat['underlying_log'] = np.log(df_feat['underlying'])
            df_feat['underlying_norm'] = df_feat['underlying'] / df_feat['underlying'].mean()
            
        # Rolling statistics on underlying
        for window in [3, 5, 10, 20]:
            if len(df_feat) > window:
                df_feat[f'underlying_ma_{window}'] = df_feat['underlying'].rolling(window, min_periods=1).mean()
                df_feat[f'underlying_std_{window}'] = df_feat['underlying'].rolling(window, min_periods=1).std().fillna(0)
                df_feat[f'underlying_ret_{window}'] = df_feat['underlying'].pct_change(window).fillna(0)
        
        # Momentum and volatility indicators
        if len(df_feat) > 1:
            df_feat['underlying_momentum'] = df_feat['underlying'].diff().fillna(0)
            df_feat['underlying_momentum_ma'] = df_feat['underlying_momentum'].rolling(5, min_periods=1).mean()
            df_feat['price_velocity'] = df_feat['underlying'].diff().rolling(3, min_periods=1).mean().fillna(0)
            df_feat['price_acceleration'] = df_feat['price_velocity'].diff().fillna(0)
        
        # Cross-features with X variables
        x_cols = [col for col in df_feat.columns if col.startswith('X')]
        
        # PCA-like features from X variables
        if len(x_cols) > 0:
            x_data = df_feat[x_cols].values
            df_feat['x_sum'] = np.sum(x_data, axis=1)
            df_feat['x_mean'] = np.mean(x_data, axis=1)
            df_feat['x_std'] = np.std(x_data, axis=1)
            df_feat['x_skew'] = [self.safe_skew(row) for row in x_data]
            df_feat['x_kurt'] = [self.safe_kurt(row) for row in x_data]
            
            # Interaction with underlying
            if 'underlying' in df_feat.columns:
                df_feat['x_underlying_corr'] = [
                    np.corrcoef(row, [df_feat['underlying'].iloc[i]])[0,1] if not np.isnan(row).all() else 0
                    for i, row in enumerate(x_data)
                ]
        
        # Time-based features if available
        if is_train and 'timestamp' in df_feat.columns:
            df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
            df_feat['hour'] = df_feat['timestamp'].dt.hour
            df_feat['minute'] = df_feat['timestamp'].dt.minute
            df_feat['second'] = df_feat['timestamp'].dt.second
            df_feat['time_numeric'] = df_feat['hour'] * 3600 + df_feat['minute'] * 60 + df_feat['second']
            
            # Cyclical time features
            df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
            df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)
            df_feat['minute_sin'] = np.sin(2 * np.pi * df_feat['minute'] / 60)
            df_feat['minute_cos'] = np.cos(2 * np.pi * df_feat['minute'] / 60)
        
        return df_feat
    
    def safe_skew(self, arr):
        """Safely calculate skewness"""
        try:
            clean_arr = arr[~np.isnan(arr)]
            if len(clean_arr) < 3:
                return 0
            return float(pd.Series(clean_arr).skew())
        except:
            return 0
    
    def safe_kurt(self, arr):
        """Safely calculate kurtosis"""
        try:
            clean_arr = arr[~np.isnan(arr)]
            if len(clean_arr) < 4:
                return 0
            return float(pd.Series(clean_arr).kurtosis())
        except:
            return 0
    
    def create_volatility_surface_features(self, df):
        """Create features based on volatility surface structure"""
        df_surf = df.copy()
        iv_cols = [col for col in df.columns if 'iv_' in col]
        
        if len(iv_cols) == 0:
            return df_surf
        
        # Separate calls and puts
        call_cols = [col for col in iv_cols if col.startswith('call_')]
        put_cols = [col for col in iv_cols if col.startswith('put_')]
        
        # Extract strikes for sorting
        def get_strike(col):
            return int(col.split('_')[-1])
        
        call_cols = sorted(call_cols, key=get_strike)
        put_cols = sorted(put_cols, key=get_strike)
        
        # Surface statistics
        if len(call_cols) > 0:
            call_ivs = df_surf[call_cols].values
            df_surf['call_iv_mean'] = np.nanmean(call_ivs, axis=1)
            df_surf['call_iv_std'] = np.nanstd(call_ivs, axis=1)
            df_surf['call_iv_min'] = np.nanmin(call_ivs, axis=1)
            df_surf['call_iv_max'] = np.nanmax(call_ivs, axis=1)
            df_surf['call_iv_range'] = df_surf['call_iv_max'] - df_surf['call_iv_min']
            
        if len(put_cols) > 0:
            put_ivs = df_surf[put_cols].values
            df_surf['put_iv_mean'] = np.nanmean(put_ivs, axis=1)
            df_surf['put_iv_std'] = np.nanstd(put_ivs, axis=1)
            df_surf['put_iv_min'] = np.nanmin(put_ivs, axis=1)
            df_surf['put_iv_max'] = np.nanmax(put_ivs, axis=1)
            df_surf['put_iv_range'] = df_surf['put_iv_max'] - df_surf['put_iv_min']
        
        # Call-Put relationships (Put-Call parity insights)
        if len(call_cols) > 0 and len(put_cols) > 0:
            # Match strikes
            call_strikes = {get_strike(col): col for col in call_cols}
            put_strikes = {get_strike(col): col for col in put_cols}
            common_strikes = set(call_strikes.keys()) & set(put_strikes.keys())
            
            if common_strikes:
                cp_diffs = []
                for strike in common_strikes:
                    call_col = call_strikes[strike]
                    put_col = put_strikes[strike]
                    diff = df_surf[call_col] - df_surf[put_col]
                    cp_diffs.append(diff)
                
                if cp_diffs:
                    cp_diff_matrix = np.column_stack(cp_diffs)
                    df_surf['cp_iv_diff_mean'] = np.nanmean(cp_diff_matrix, axis=1)
                    df_surf['cp_iv_diff_std'] = np.nanstd(cp_diff_matrix, axis=1)
        
        return df_surf
    
    def interpolate_missing_ivs(self, df, iv_columns):
        """Advanced interpolation for missing IVs using volatility surface constraints"""
        df_interp = df.copy()
        
        for idx in df_interp.index:
            row = df_interp.loc[idx, iv_columns].copy()
            
            if row.isna().any() and not row.isna().all():
                # Separate calls and puts
                call_data = {}
                put_data = {}
                
                for col in iv_columns:
                    strike = int(col.split('_')[-1])
                    if col.startswith('call_'):
                        call_data[strike] = row[col]
                    elif col.startswith('put_'):
                        put_data[strike] = row[col]
                
                # Interpolate calls
                if call_data:
                    strikes = sorted(call_data.keys())
                    values = [call_data[s] for s in strikes]
                    valid_mask = ~pd.isna(values)
                    
                    if valid_mask.sum() > 1:
                        valid_strikes = np.array(strikes)[valid_mask]
                        valid_values = np.array(values)[valid_mask]
                        
                        # Use spline interpolation
                        try:
                            spline = UnivariateSpline(valid_strikes, valid_values, s=0, ext=3)
                            for i, strike in enumerate(strikes):
                                if pd.isna(values[i]):
                                    interpolated_val = spline(strike)
                                    # Ensure reasonable bounds
                                    interpolated_val = max(0.01, min(2.0, interpolated_val))
                                    df_interp.loc[idx, f'call_iv_{strike}'] = interpolated_val
                        except:
                            # Fallback to linear interpolation
                            if len(valid_strikes) >= 2:
                                interp_func = interp1d(valid_strikes, valid_values, 
                                                     kind='linear', fill_value='extrapolate')
                                for i, strike in enumerate(strikes):
                                    if pd.isna(values[i]):
                                        interpolated_val = interp_func(strike)
                                        interpolated_val = max(0.01, min(2.0, interpolated_val))
                                        df_interp.loc[idx, f'call_iv_{strike}'] = interpolated_val
                
                # Interpolate puts similarly
                if put_data:
                    strikes = sorted(put_data.keys())
                    values = [put_data[s] for s in strikes]
                    valid_mask = ~pd.isna(values)
                    
                    if valid_mask.sum() > 1:
                        valid_strikes = np.array(strikes)[valid_mask]
                        valid_values = np.array(values)[valid_mask]
                        
                        try:
                            spline = UnivariateSpline(valid_strikes, valid_values, s=0, ext=3)
                            for i, strike in enumerate(strikes):
                                if pd.isna(values[i]):
                                    interpolated_val = spline(strike)
                                    interpolated_val = max(0.01, min(2.0, interpolated_val))
                                    df_interp.loc[idx, f'put_iv_{strike}'] = interpolated_val
                        except:
                            if len(valid_strikes) >= 2:
                                interp_func = interp1d(valid_strikes, valid_values, 
                                                     kind='linear', fill_value='extrapolate')
                                for i, strike in enumerate(strikes):
                                    if pd.isna(values[i]):
                                        interpolated_val = interp_func(strike)
                                        interpolated_val = max(0.01, min(2.0, interpolated_val))
                                        df_interp.loc[idx, f'put_iv_{strike}'] = interpolated_val
        
        return df_interp
    
    def create_ensemble_models(self):
        """Create diverse ensemble of models"""
        models = {
            'lgb1': LGBMRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=6,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            ),
            'lgb2': LGBMRegressor(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=4,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.05,
                reg_lambda=0.05,
                random_state=123,
                verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.015,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'et': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'knn': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                metric='minkowski'
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        return models
    
    def fit_predict_individual_models(self, X_train, y_train, X_test):
        """Fit individual models for each IV column"""
        models = self.create_ensemble_models()
        predictions = {}
        
        for col in self.iv_columns:
            print(f"Training models for {col}...")
            
            # Get non-null targets for this column
            valid_idx = ~y_train[col].isna()
            if valid_idx.sum() == 0:
                # No valid data for this column, use mean prediction
                predictions[col] = np.full(len(X_test), y_train[col].mean())
                continue
            
            X_col = X_train[valid_idx]
            y_col = y_train.loc[valid_idx, col]
            
            col_predictions = []
            
            for model_name, model in models.items():
                try:
                    model.fit(X_col, y_col)
                    pred = model.predict(X_test)
                    # Ensure reasonable bounds for IV
                    pred = np.clip(pred, 0.01, 2.0)
                    col_predictions.append(pred)
                except Exception as e:
                    print(f"Error with {model_name} for {col}: {e}")
                    # Fallback prediction
                    col_predictions.append(np.full(len(X_test), y_col.mean()))
            
            # Ensemble prediction (weighted average)
            weights = [0.25, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05]  # Favor gradient boosting
            if len(col_predictions) == len(weights):
                ensemble_pred = np.average(col_predictions, axis=0, weights=weights)
            else:
                ensemble_pred = np.mean(col_predictions, axis=0)
            
            predictions[col] = ensemble_pred
        
        return predictions

def main():
    # File paths
    train_path = "train_data.parquet"
    test_path = "test_data.parquet"
    sample_path = "sample_submission.csv"
    output_path = "advanced_optimized_submission.csv"
    
    print("Loading data...")
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    sample = pd.read_csv(sample_path)
    
    # Initialize predictor
    predictor = AdvancedIVPredictor()
    predictor.iv_columns = [col for col in sample.columns if col != 'timestamp' and col in train.columns]
    
    print(f"Found {len(predictor.iv_columns)} IV columns to predict")
    
    # Advanced feature engineering
    print("Creating advanced features...")
    train_feat = predictor.advanced_feature_engineering(train, is_train=True)
    test_feat = predictor.advanced_feature_engineering(test, is_train=False)
    
    # Add volatility surface features
    print("Creating volatility surface features...")
    train_feat = predictor.create_volatility_surface_features(train_feat)
    
    # Interpolate missing training IVs for better surface features
    print("Interpolating missing training IVs...")
    train_feat = predictor.interpolate_missing_ivs(train_feat, predictor.iv_columns)
    
    # Select features (exclude timestamp and IV columns)
    feature_cols = [col for col in train_feat.columns 
                   if col not in predictor.iv_columns and col != 'timestamp' and col != 'expiry']
    
    # Handle missing values in features
    print("Handling missing values...")
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(
        imputer.fit_transform(train_feat[feature_cols]),
        columns=feature_cols,
        index=train_feat.index
    )
    X_test = pd.DataFrame(
        imputer.transform(test_feat[feature_cols]),
        columns=feature_cols,
        index=test_feat.index
    )
    
    # Scale features
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )
    
    # Prepare targets
    y_train = train_feat[predictor.iv_columns]
    
    # Fit models and make predictions
    print("Training ensemble models...")
    predictions = predictor.fit_predict_individual_models(X_train_scaled, y_train, X_test_scaled)
    
    # Create submission
    print("Creating submission...")
    submission = sample.copy()
    
    for col in predictor.iv_columns:
        if col in predictions:
            # Only fill NaN values in test data
            mask = test[col].isna()
            submission.loc[mask, col] = predictions[col][mask]
    
    # Post-processing: ensure no NaN values remain
    for col in predictor.iv_columns:
        if submission[col].isna().any():
            # Fill remaining NaN with column mean from training data
            fill_value = train[col].mean() if not pd.isna(train[col].mean()) else 0.2
            submission[col].fillna(fill_value, inplace=True)
    
    # Final bounds check
    for col in predictor.iv_columns:
        submission[col] = np.clip(submission[col], 0.01, 2.0)
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"Advanced submission saved to: {output_path}")
    print(f"Submission shape: {submission.shape}")
    print("Sample predictions:")
    print(submission[predictor.iv_columns].head())

if __name__ == "__main__":
    main()