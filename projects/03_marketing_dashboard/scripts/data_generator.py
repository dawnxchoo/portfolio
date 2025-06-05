"""
Generate synthetic marketing campaign data for dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MarketingDataGenerator:
    def __init__(self, start_date='2023-01-01', end_date='2023-12-31', seed=42):
        np.random.seed(seed)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self.channels = ['Email', 'Social Media', 'Search', 'Display', 'Affiliate', 'Content Marketing', 'Video']
        self.campaigns = [
            'Brand Awareness Q1', 'Spring Sale 2023', 'Summer Campaign 2023',
            'Back to School 2023', 'Black Friday 2023', 'Holiday Season 2023',
            'New Year 2024', 'Valentine\'s Day 2024'
        ]
        self.segments = ['New Customers', 'Returning Customers', 'VIP Customers', 'At-Risk', 'Dormant']
        self.devices = ['Mobile', 'Desktop', 'Tablet']
        self.regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
        
    def generate_base_metrics(self, channel):
        """Generate base metrics for each channel"""
        channel_profiles = {
            'Email': {
                'spend_range': (500, 2000),
                'ctr_range': (0.02, 0.05),
                'conversion_rate': (0.03, 0.06),
                'avg_order_value': (80, 120)
            },
            'Social Media': {
                'spend_range': (1500, 4000),
                'ctr_range': (0.01, 0.03),
                'conversion_rate': (0.02, 0.04),
                'avg_order_value': (60, 100)
            },
            'Search': {
                'spend_range': (3000, 8000),
                'ctr_range': (0.03, 0.07),
                'conversion_rate': (0.04, 0.08),
                'avg_order_value': (100, 150)
            },
            'Display': {
                'spend_range': (1000, 3000),
                'ctr_range': (0.005, 0.02),
                'conversion_rate': (0.01, 0.03),
                'avg_order_value': (70, 110)
            },
            'Affiliate': {
                'spend_range': (500, 2000),
                'ctr_range': (0.02, 0.05),
                'conversion_rate': (0.05, 0.10),
                'avg_order_value': (90, 140)
            },
            'Content Marketing': {
                'spend_range': (800, 2500),
                'ctr_range': (0.015, 0.04),
                'conversion_rate': (0.03, 0.06),
                'avg_order_value': (85, 125)
            },
            'Video': {
                'spend_range': (2000, 5000),
                'ctr_range': (0.02, 0.04),
                'conversion_rate': (0.02, 0.05),
                'avg_order_value': (75, 115)
            }
        }
        
        return channel_profiles.get(channel, channel_profiles['Display'])
    
    def apply_seasonality(self, date, base_value):
        """Apply seasonal multipliers to base values"""
        month = date.month
        day_of_week = date.dayofweek
        
        # Monthly seasonality
        monthly_multipliers = {
            1: 0.8,   # January - post-holiday slowdown
            2: 0.85,  # February
            3: 0.9,   # March
            4: 0.95,  # April
            5: 1.0,   # May
            6: 1.1,   # June - summer begins
            7: 1.15,  # July
            8: 1.1,   # August
            9: 1.2,   # September - back to school
            10: 1.15, # October
            11: 1.5,  # November - Black Friday
            12: 1.8   # December - holiday season
        }
        
        # Day of week multipliers (0 = Monday, 6 = Sunday)
        dow_multipliers = {
            0: 1.1,   # Monday
            1: 1.05,  # Tuesday
            2: 1.0,   # Wednesday
            3: 1.0,   # Thursday
            4: 1.15,  # Friday
            5: 0.9,   # Saturday
            6: 0.85   # Sunday
        }
        
        seasonal_value = base_value * monthly_multipliers[month] * dow_multipliers[day_of_week]
        
        # Add some random variation
        return seasonal_value * np.random.uniform(0.9, 1.1)
    
    def assign_campaign(self, date):
        """Assign campaign based on date"""
        month = date.month
        
        if month in [1, 2]:
            return np.random.choice(['Brand Awareness Q1', 'Valentine\'s Day 2024'])
        elif month in [3, 4, 5]:
            return 'Spring Sale 2023'
        elif month in [6, 7]:
            return 'Summer Campaign 2023'
        elif month in [8, 9]:
            return 'Back to School 2023'
        elif month == 11:
            return 'Black Friday 2023'
        elif month == 12:
            return 'Holiday Season 2023'
        else:
            return 'Brand Awareness Q1'
    
    def generate_daily_data(self):
        """Generate daily marketing data"""
        data = []
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        for date in dates:
            for channel in self.channels:
                profile = self.generate_base_metrics(channel)
                
                # Base metrics
                base_spend = np.random.uniform(*profile['spend_range'])
                base_ctr = np.random.uniform(*profile['ctr_range'])
                base_conversion_rate = np.random.uniform(*profile['conversion_rate'])
                base_aov = np.random.uniform(*profile['avg_order_value'])
                
                # Apply seasonality
                spend = self.apply_seasonality(date, base_spend)
                
                # Calculate other metrics
                impressions = int(spend * np.random.uniform(200, 500))
                clicks = int(impressions * base_ctr)
                conversions = int(clicks * base_conversion_rate)
                revenue = conversions * base_aov
                
                # Additional metrics
                new_customers = int(conversions * np.random.uniform(0.3, 0.5))
                returning_customers = conversions - new_customers
                
                # Device and region distribution
                mobile_share = np.random.uniform(0.5, 0.7)
                desktop_share = np.random.uniform(0.2, 0.4)
                tablet_share = 1 - mobile_share - desktop_share
                
                data.append({
                    'date': date,
                    'channel': channel,
                    'campaign': self.assign_campaign(date),
                    'spend': spend,
                    'impressions': impressions,
                    'clicks': clicks,
                    'conversions': conversions,
                    'revenue': revenue,
                    'new_customers': new_customers,
                    'returning_customers': returning_customers,
                    'mobile_conversions': int(conversions * mobile_share),
                    'desktop_conversions': int(conversions * desktop_share),
                    'tablet_conversions': int(conversions * tablet_share),
                    'bounce_rate': np.random.uniform(0.3, 0.7),
                    'avg_session_duration': np.random.uniform(60, 300),
                    'pages_per_session': np.random.uniform(2, 6)
                })
        
        return pd.DataFrame(data)
    
    def generate_customer_data(self):
        """Generate customer-level data"""
        n_customers = 10000
        
        customer_data = {
            'customer_id': range(1, n_customers + 1),
            'acquisition_date': pd.to_datetime('2023-01-01') + pd.to_timedelta(
                np.random.randint(0, 365, n_customers), unit='D'
            ),
            'segment': np.random.choice(self.segments, n_customers, 
                                      p=[0.4, 0.3, 0.1, 0.15, 0.05]),
            'lifetime_value': np.random.exponential(scale=200, size=n_customers),
            'total_orders': np.random.poisson(lam=3, size=n_customers),
            'preferred_channel': np.random.choice(self.channels, n_customers),
            'churn_risk_score': np.random.uniform(0, 1, n_customers)
        }
        
        return pd.DataFrame(customer_data)
    
    def generate_ab_test_data(self):
        """Generate A/B test results"""
        tests = [
            {
                'test_name': 'Email Subject Line Test',
                'test_type': 'Email',
                'start_date': '2023-06-01',
                'end_date': '2023-06-14',
                'variant_a_name': 'Discount Focus',
                'variant_b_name': 'Urgency Focus',
                'variant_a_conversions': 245,
                'variant_b_conversions': 312,
                'variant_a_sample_size': 5000,
                'variant_b_sample_size': 5000,
                'statistical_significance': 0.98,
                'lift': 0.27
            },
            {
                'test_name': 'Landing Page CTA Color',
                'test_type': 'Website',
                'start_date': '2023-07-15',
                'end_date': '2023-07-28',
                'variant_a_name': 'Blue Button',
                'variant_b_name': 'Orange Button',
                'variant_a_conversions': 189,
                'variant_b_conversions': 201,
                'variant_a_sample_size': 4000,
                'variant_b_sample_size': 4000,
                'statistical_significance': 0.72,
                'lift': 0.06
            },
            {
                'test_name': 'Ad Copy Variation',
                'test_type': 'Search',
                'start_date': '2023-09-01',
                'end_date': '2023-09-14',
                'variant_a_name': 'Feature Focus',
                'variant_b_name': 'Benefit Focus',
                'variant_a_conversions': 423,
                'variant_b_conversions': 567,
                'variant_a_sample_size': 8000,
                'variant_b_sample_size': 8000,
                'statistical_significance': 0.99,
                'lift': 0.34
            }
        ]
        
        return pd.DataFrame(tests)
    
    def save_data(self, output_dir='data'):
        """Save all generated data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save daily metrics
        daily_data = self.generate_daily_data()
        daily_data.to_csv(os.path.join(output_dir, 'daily_marketing_metrics.csv'), index=False)
        
        # Generate and save customer data
        customer_data = self.generate_customer_data()
        customer_data.to_csv(os.path.join(output_dir, 'customer_data.csv'), index=False)
        
        # Generate and save A/B test data
        ab_test_data = self.generate_ab_test_data()
        ab_test_data.to_csv(os.path.join(output_dir, 'ab_test_results.csv'), index=False)
        
        print(f"Data generated and saved to {output_dir}/")
        print(f"- daily_marketing_metrics.csv: {len(daily_data)} records")
        print(f"- customer_data.csv: {len(customer_data)} records")
        print(f"- ab_test_results.csv: {len(ab_test_data)} records")

if __name__ == "__main__":
    generator = MarketingDataGenerator()
    generator.save_data()