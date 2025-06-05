"""
Sales Forecasting Model Training Pipeline
Production-ready script for automated model training and deployment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
from typing import Dict, Tuple, Any
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SalesForecastingPipeline:
    """End-to-end pipeline for sales forecasting model training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metrics = {}
        
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load sales data from database"""
        logger.info(f"Loading data from {start_date} to {end_date}")
        
        # In production, this would connect to your data warehouse
        # For demo, using synthetic data generation
        query = f"""
        SELECT 
            date,
            store_id,
            sales,
            is_weekend,
            is_holiday,
            promotion,
            temperature,
            competitor_promo
        FROM sales_data
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY store_id, date
        """
        
        # Simulated data loading
        df = self._generate_synthetic_data(start_date, end_date)
        logger.info(f"Loaded {len(df):,} records")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series features"""
        logger.info("Creating features...")
        
        df = df.copy()
        df = df.sort_values(['store_id', 'date'])
        
        # Date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['dayofyear'] = df['date'].dt.dayofyear
        
        # Lag features
        for lag in self.config['lag_features']:
            df[f'sales_lag_{lag}'] = df.groupby('store_id')['sales'].shift(lag)
        
        # Rolling statistics
        for window in self.config['rolling_windows']:
            df[f'sales_roll_mean_{window}'] = df.groupby('store_id')['sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'sales_roll_std_{window}'] = df.groupby('store_id')['sales'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        
        # Store-level features
        store_features = df.groupby('store_id')['sales'].agg(['mean', 'std'])
        store_features.columns = ['store_avg_sales', 'store_std_sales']
        df = df.merge(store_features, on='store_id', how='left')
        
        logger.info(f"Created {len(df.columns)} features")
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and validation sets"""
        df = df.dropna()
        
        # Time-based split
        split_date = df['date'].max() - timedelta(days=self.config['validation_days'])
        train_mask = df['date'] < split_date
        val_mask = df['date'] >= split_date
        
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'sales', 'store_id']]
        self.feature_columns = feature_cols
        
        X_train = df[train_mask][feature_cols]
        X_val = df[val_mask][feature_cols]
        y_train = df[train_mask]['sales']
        y_val = df[val_mask]['sales']
        
        logger.info(f"Train size: {len(X_train):,}, Validation size: {len(X_val):,}")
        return X_train, X_val, y_train, y_val
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model"""
        logger.info("Training model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = xgb.XGBRegressor(**self.config['model_params'])
        self.model.fit(X_train_scaled, y_train)
        
        logger.info("Model training completed")
    
    def evaluate_model(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        # Make predictions
        X_val_scaled = self.scaler.transform(X_val)
        y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        r2 = r2_score(y_val, y_pred)
        
        self.metrics = {
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'validation_samples': len(y_val)
        }
        
        logger.info(f"Model Performance - MAPE: {mape:.2f}%, R²: {r2:.4f}")
        return self.metrics
    
    def save_model(self, path: str) -> None:
        """Save model artifacts"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_artifacts, f)
        
        logger.info(f"Model saved to {path}")
    
    def run_pipeline(self) -> None:
        """Execute full training pipeline"""
        logger.info("Starting training pipeline...")
        
        # Load data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=self.config['training_days'])).strftime('%Y-%m-%d')
        df = self.load_data(start_date, end_date)
        
        # Create features
        df = self.create_features(df)
        
        # Split data
        X_train, X_val, y_train, y_val = self.split_data(df)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        self.evaluate_model(X_val, y_val)
        
        # Save model
        model_path = f"models/sales_forecast_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.save_model(model_path)
        
        logger.info("Pipeline completed successfully!")
    
    def _generate_synthetic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate synthetic data for demo purposes"""
        dates = pd.date_range(start_date, end_date, freq='D')
        data = []
        
        for store_id in range(1, 51):
            base_sales = np.random.uniform(5000, 20000)
            
            for date in dates:
                sales = base_sales * (1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365))
                sales *= np.random.uniform(0.8, 1.2)
                
                data.append({
                    'date': date,
                    'store_id': store_id,
                    'sales': sales,
                    'is_weekend': int(date.dayofweek in [5, 6]),
                    'is_holiday': int(date.month == 12 and date.day > 15),
                    'promotion': np.random.choice([0, 1], p=[0.85, 0.15]),
                    'temperature': np.random.normal(20, 10),
                    'competitor_promo': np.random.choice([0, 1], p=[0.8, 0.2])
                })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Configuration
    config = {
        'training_days': 730,  # 2 years of data
        'validation_days': 90,  # 3 months validation
        'lag_features': [1, 7, 14, 30],
        'rolling_windows': [7, 14, 30],
        'model_params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
    
    # Run pipeline
    pipeline = SalesForecastingPipeline(config)
    pipeline.run_pipeline()