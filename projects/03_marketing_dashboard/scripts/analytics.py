"""
Marketing analytics calculations and metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional

class MarketingAnalytics:
    def __init__(self):
        self.metrics_definitions = {
            'ctr': 'Click-Through Rate',
            'cpc': 'Cost Per Click',
            'cpa': 'Cost Per Acquisition',
            'roas': 'Return on Ad Spend',
            'ltv': 'Customer Lifetime Value',
            'cac': 'Customer Acquisition Cost',
            'aov': 'Average Order Value'
        }
    
    @staticmethod
    def calculate_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic marketing metrics"""
        df = df.copy()
        
        # Click-through rate
        df['ctr'] = np.where(df['impressions'] > 0, 
                            df['clicks'] / df['impressions'], 0)
        
        # Cost per click
        df['cpc'] = np.where(df['clicks'] > 0, 
                            df['spend'] / df['clicks'], 0)
        
        # Cost per acquisition
        df['cpa'] = np.where(df['conversions'] > 0, 
                            df['spend'] / df['conversions'], 0)
        
        # Return on ad spend
        df['roas'] = np.where(df['spend'] > 0, 
                             df['revenue'] / df['spend'], 0)
        
        # Conversion rate
        df['conversion_rate'] = np.where(df['clicks'] > 0, 
                                        df['conversions'] / df['clicks'], 0)
        
        return df
    
    @staticmethod
    def calculate_attribution(df: pd.DataFrame, model: str = 'last_touch') -> pd.DataFrame:
        """Calculate channel attribution based on different models"""
        if model == 'last_touch':
            # Simple last-touch attribution
            attribution = df.groupby('channel').agg({
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
        elif model == 'linear':
            # Linear attribution (equal credit to all touchpoints)
            # Simplified version - in production would track actual user journeys
            attribution = df.groupby('channel').agg({
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            # Normalize by estimated touchpoints
            attribution['conversions'] = attribution['conversions'] * 0.7
            attribution['revenue'] = attribution['revenue'] * 0.7
            
        elif model == 'time_decay':
            # Time decay attribution (more credit to recent touchpoints)
            df_sorted = df.sort_values('date', ascending=False)
            decay_factor = 0.9
            
            attribution_data = []
            for channel in df['channel'].unique():
                channel_data = df_sorted[df_sorted['channel'] == channel].copy()
                channel_data['weight'] = decay_factor ** np.arange(len(channel_data))
                
                weighted_conversions = (channel_data['conversions'] * channel_data['weight']).sum()
                weighted_revenue = (channel_data['revenue'] * channel_data['weight']).sum()
                
                attribution_data.append({
                    'channel': channel,
                    'conversions': weighted_conversions,
                    'revenue': weighted_revenue
                })
            
            attribution = pd.DataFrame(attribution_data)
        
        else:
            raise ValueError(f"Unknown attribution model: {model}")
        
        # Calculate percentages
        attribution['conversion_share'] = (attribution['conversions'] / 
                                         attribution['conversions'].sum() * 100)
        attribution['revenue_share'] = (attribution['revenue'] / 
                                       attribution['revenue'].sum() * 100)
        
        return attribution
    
    @staticmethod
    def calculate_ltv(customer_df: pd.DataFrame, 
                     time_period: int = 12) -> pd.DataFrame:
        """Calculate customer lifetime value"""
        ltv_data = []
        
        for segment in customer_df['segment'].unique():
            segment_data = customer_df[customer_df['segment'] == segment]
            
            # Average purchase value
            avg_purchase_value = segment_data['lifetime_value'].mean() / \
                               segment_data['total_orders'].replace(0, 1).mean()
            
            # Purchase frequency (orders per period)
            purchase_frequency = segment_data['total_orders'].mean() / time_period
            
            # Customer lifespan (simplified - would use churn analysis in production)
            avg_lifespan = 24  # months
            
            # Calculate LTV
            ltv = avg_purchase_value * purchase_frequency * avg_lifespan
            
            ltv_data.append({
                'segment': segment,
                'avg_purchase_value': avg_purchase_value,
                'purchase_frequency': purchase_frequency,
                'avg_lifespan_months': avg_lifespan,
                'ltv': ltv,
                'customer_count': len(segment_data)
            })
        
        return pd.DataFrame(ltv_data)
    
    @staticmethod
    def ab_test_analysis(variant_a_conversions: int, variant_a_visitors: int,
                        variant_b_conversions: int, variant_b_visitors: int,
                        confidence_level: float = 0.95) -> Dict:
        """Perform statistical analysis on A/B test results"""
        # Conversion rates
        rate_a = variant_a_conversions / variant_a_visitors
        rate_b = variant_b_conversions / variant_b_visitors
        
        # Pooled probability
        p_pool = (variant_a_conversions + variant_b_conversions) / \
                (variant_a_visitors + variant_b_visitors)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * 
                    (1/variant_a_visitors + 1/variant_b_visitors))
        
        # Z-score
        z_score = (rate_b - rate_a) / se
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_critical * se
        ci_lower = (rate_b - rate_a) - margin_of_error
        ci_upper = (rate_b - rate_a) + margin_of_error
        
        # Lift
        lift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0
        
        return {
            'variant_a_rate': rate_a,
            'variant_b_rate': rate_b,
            'lift': lift,
            'p_value': p_value,
            'is_significant': p_value < (1 - confidence_level),
            'confidence_interval': (ci_lower, ci_upper),
            'z_score': z_score
        }
    
    @staticmethod
    def cohort_analysis(df: pd.DataFrame, 
                       date_col: str = 'date',
                       metric_col: str = 'revenue') -> pd.DataFrame:
        """Perform cohort analysis"""
        df = df.copy()
        
        # Create cohort month
        df['cohort_month'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        df['transaction_month'] = pd.to_datetime(df[date_col]).dt.to_period('M')
        
        # Calculate months since first purchase
        df['months_since_first'] = (df['transaction_month'] - df['cohort_month']).apply(lambda x: x.n)
        
        # Aggregate by cohort and months since first purchase
        cohort_data = df.groupby(['cohort_month', 'months_since_first'])[metric_col].agg(['sum', 'count'])
        cohort_data = cohort_data.reset_index()
        
        # Pivot for visualization
        cohort_pivot = cohort_data.pivot(index='cohort_month', 
                                        columns='months_since_first', 
                                        values='sum')
        
        # Calculate retention rates (percentage of initial cohort value)
        cohort_sizes = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100
        
        return retention_matrix
    
    @staticmethod
    def calculate_marketing_mix_model(df: pd.DataFrame) -> Dict:
        """Simple marketing mix modeling"""
        # This is a simplified version - production would use more sophisticated models
        
        # Calculate correlations between spend and revenue by channel
        channel_effectiveness = {}
        
        for channel in df['channel'].unique():
            channel_data = df[df['channel'] == channel]
            
            # Calculate elasticity (% change in revenue / % change in spend)
            spend_pct_change = channel_data['spend'].pct_change()
            revenue_pct_change = channel_data['revenue'].pct_change()
            
            # Remove infinite values
            mask = np.isfinite(spend_pct_change) & np.isfinite(revenue_pct_change)
            elasticity = (revenue_pct_change[mask] / spend_pct_change[mask]).mean()
            
            # Calculate saturation point (simplified)
            # In production, would fit a response curve
            max_efficiency_spend = channel_data.groupby('spend')['roas'].mean().idxmax()
            
            channel_effectiveness[channel] = {
                'elasticity': elasticity if np.isfinite(elasticity) else 0,
                'avg_roas': channel_data['roas'].mean(),
                'saturation_point': max_efficiency_spend,
                'current_spend': channel_data['spend'].mean()
            }
        
        return channel_effectiveness
    
    @staticmethod
    def forecast_performance(df: pd.DataFrame, 
                           periods: int = 30,
                           method: str = 'moving_average') -> pd.DataFrame:
        """Forecast future performance"""
        # Simple forecasting - production would use ARIMA, Prophet, etc.
        
        forecast_data = []
        
        for channel in df['channel'].unique():
            channel_data = df[df['channel'] == channel].sort_values('date')
            
            if method == 'moving_average':
                # 7-day moving average
                ma_window = min(7, len(channel_data))
                recent_metrics = channel_data.tail(ma_window).mean()
                
                # Add trend component
                if len(channel_data) > 14:
                    trend = channel_data.tail(14)['revenue'].pct_change().mean()
                else:
                    trend = 0
                
                for i in range(periods):
                    forecast_date = channel_data['date'].max() + pd.Timedelta(days=i+1)
                    
                    forecast_data.append({
                        'date': forecast_date,
                        'channel': channel,
                        'forecast_spend': recent_metrics['spend'] * (1 + trend * i/periods),
                        'forecast_revenue': recent_metrics['revenue'] * (1 + trend * i/periods),
                        'forecast_conversions': int(recent_metrics['conversions'] * (1 + trend * i/periods)),
                        'confidence_lower': recent_metrics['revenue'] * (1 + trend * i/periods) * 0.8,
                        'confidence_upper': recent_metrics['revenue'] * (1 + trend * i/periods) * 1.2
                    })
        
        return pd.DataFrame(forecast_data)
    
    @staticmethod
    def customer_segmentation_analysis(customer_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze customer segments"""
        segment_analysis = customer_df.groupby('segment').agg({
            'customer_id': 'count',
            'lifetime_value': ['mean', 'sum'],
            'total_orders': 'mean',
            'churn_risk_score': 'mean'
        }).round(2)
        
        segment_analysis.columns = ['customer_count', 'avg_ltv', 'total_ltv', 
                                   'avg_orders', 'avg_churn_risk']
        
        # Calculate segment value contribution
        segment_analysis['value_contribution'] = (segment_analysis['total_ltv'] / 
                                                 segment_analysis['total_ltv'].sum() * 100)
        
        return segment_analysis.reset_index()

def generate_marketing_insights(daily_df: pd.DataFrame, 
                              customer_df: pd.DataFrame) -> Dict:
    """Generate key marketing insights"""
    analytics = MarketingAnalytics()
    
    # Calculate basic metrics
    daily_df = analytics.calculate_basic_metrics(daily_df)
    
    insights = {
        'top_performing_channel': daily_df.groupby('channel')['roas'].mean().idxmax(),
        'worst_performing_channel': daily_df.groupby('channel')['roas'].mean().idxmin(),
        'avg_cac': daily_df['spend'].sum() / daily_df['new_customers'].sum(),
        'total_roi': ((daily_df['revenue'].sum() - daily_df['spend'].sum()) / 
                     daily_df['spend'].sum() * 100),
        'best_campaign': daily_df.groupby('campaign')['roas'].mean().idxmax(),
        'mobile_revenue_share': (daily_df['mobile_conversions'].sum() / 
                               daily_df['conversions'].sum() * 100)
    }
    
    # Segment insights
    segment_analysis = analytics.customer_segmentation_analysis(customer_df)
    insights['most_valuable_segment'] = segment_analysis.loc[
        segment_analysis['avg_ltv'].idxmax(), 'segment'
    ]
    insights['highest_churn_risk_segment'] = segment_analysis.loc[
        segment_analysis['avg_churn_risk'].idxmax(), 'segment'
    ]
    
    return insights