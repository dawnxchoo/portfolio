# Sales Forecasting Model - Machine Learning Project

## Project Overview
This project develops a machine learning model to forecast monthly sales for a retail chain with 50 stores. Using advanced time series techniques and external features, the model achieves 92% accuracy and helps optimize inventory management, reducing stockouts by 35%.

## Business Impact
- **Problem**: Inaccurate sales forecasts leading to $1.2M annual loss from stockouts and overstock
- **Solution**: ML-powered forecasting system with store-specific predictions
- **Result**: 35% reduction in stockouts, 20% reduction in excess inventory, $800K annual savings

## Model Performance
- **MAPE (Mean Absolute Percentage Error)**: 8.2%
- **R² Score**: 0.92
- **Forecast Horizon**: 3 months
- **Update Frequency**: Weekly retraining

## Technical Approach
1. **Feature Engineering**: Created 47 features including lag variables, rolling statistics, and external factors
2. **Model Selection**: Tested 5 algorithms; XGBoost performed best
3. **Hyperparameter Tuning**: Bayesian optimization for optimal parameters
4. **Validation Strategy**: Time-series cross-validation with expanding window

## Key Features Used
- Historical sales (lags, rolling means)
- Seasonality indicators
- Promotional calendars
- Local economic indicators
- Weather data
- Competitor presence

## Skills Demonstrated
- Python (scikit-learn, XGBoost, statsmodels, prophet)
- Time series analysis and forecasting
- Feature engineering
- Model evaluation and selection
- MLOps practices
- Business value quantification

## Project Structure
- `notebooks/sales_forecasting_model.ipynb` - Main modeling notebook
- `scripts/model_training.py` - Production training pipeline
- `scripts/feature_engineering.py` - Feature creation module
- `data/` - Sample sales data and external features
- `models/` - Saved model artifacts
- `results/` - Performance metrics and visualizations