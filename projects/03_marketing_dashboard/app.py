"""
Marketing Campaign Analytics Dashboard
Interactive Streamlit app for marketing performance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        background-color: #e3f2fd;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Marketing Campaign Analytics Dashboard")
st.markdown("**Real-time insights for data-driven marketing decisions**")

# Sidebar for filters
st.sidebar.header("Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=90), datetime.now()),
    max_value=datetime.now()
)

# Channel filter
channels = ['All', 'Email', 'Social Media', 'Search', 'Display', 'Affiliate']
selected_channel = st.sidebar.selectbox("Select Channel", channels)

# Campaign filter
campaigns = ['All Campaigns', 'Summer Sale 2023', 'Black Friday 2023', 'New Year 2024', 'Spring Launch 2024']
selected_campaign = st.sidebar.selectbox("Select Campaign", campaigns)

# Data generation function
@st.cache_data
def load_data():
    """Generate synthetic marketing data"""
    np.random.seed(42)
    
    # Generate date range
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Channels and campaigns
    channels = ['Email', 'Social Media', 'Search', 'Display', 'Affiliate']
    campaigns = ['Summer Sale 2023', 'Black Friday 2023', 'New Year 2024', 'Spring Launch 2024']
    
    data = []
    for date in dates:
        for channel in channels:
            # Base metrics by channel
            if channel == 'Email':
                base_spend = np.random.uniform(500, 1500)
                base_conversions = np.random.poisson(50)
                base_ctr = np.random.uniform(0.02, 0.05)
            elif channel == 'Social Media':
                base_spend = np.random.uniform(1000, 3000)
                base_conversions = np.random.poisson(40)
                base_ctr = np.random.uniform(0.01, 0.03)
            elif channel == 'Search':
                base_spend = np.random.uniform(2000, 5000)
                base_conversions = np.random.poisson(80)
                base_ctr = np.random.uniform(0.03, 0.06)
            elif channel == 'Display':
                base_spend = np.random.uniform(800, 2000)
                base_conversions = np.random.poisson(30)
                base_ctr = np.random.uniform(0.005, 0.02)
            else:  # Affiliate
                base_spend = np.random.uniform(300, 1000)
                base_conversions = np.random.poisson(20)
                base_ctr = np.random.uniform(0.01, 0.04)
            
            # Seasonality
            if date.month in [11, 12]:  # Holiday season
                base_spend *= 1.5
                base_conversions = int(base_conversions * 1.8)
            
            # Campaign assignment
            if date.month in [6, 7]:
                campaign = 'Summer Sale 2023'
            elif date.month == 11:
                campaign = 'Black Friday 2023'
            elif date.month == 12:
                campaign = 'New Year 2024'
            else:
                campaign = 'Spring Launch 2024'
            
            impressions = int(base_spend * np.random.uniform(100, 200))
            clicks = int(impressions * base_ctr)
            
            data.append({
                'date': date,
                'channel': channel,
                'campaign': campaign,
                'spend': base_spend,
                'impressions': impressions,
                'clicks': clicks,
                'conversions': base_conversions,
                'revenue': base_conversions * np.random.uniform(50, 150)
            })
    
    df = pd.DataFrame(data)
    
    # Calculate derived metrics
    df['ctr'] = df['clicks'] / df['impressions']
    df['conversion_rate'] = df['conversions'] / df['clicks']
    df['cpc'] = df['spend'] / df['clicks']
    df['cpa'] = df['spend'] / df['conversions'].replace(0, 1)
    df['roas'] = df['revenue'] / df['spend']
    
    return df

# Load data
df = load_data()

# Apply filters
if selected_channel != 'All':
    df = df[df['channel'] == selected_channel]
if selected_campaign != 'All Campaigns':
    df = df[df['campaign'] == selected_campaign]

# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Channel Performance", "👥 Customer Insights", "🔮 Predictions"])

with tab1:
    st.header("Campaign Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = df['spend'].sum()
    total_revenue = df['revenue'].sum()
    total_conversions = df['conversions'].sum()
    avg_roas = df['roas'].mean()
    
    col1.metric("Total Spend", f"${total_spend:,.0f}", "↑ 12% vs last period")
    col2.metric("Total Revenue", f"${total_revenue:,.0f}", "↑ 25% vs last period")
    col3.metric("Conversions", f"{total_conversions:,}", "↑ 18% vs last period")
    col4.metric("Avg ROAS", f"{avg_roas:.2f}x", "↑ 0.3x vs last period")
    
    # Performance over time
    st.subheader("Performance Trends")
    
    daily_metrics = df.groupby('date').agg({
        'spend': 'sum',
        'revenue': 'sum',
        'conversions': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_metrics['date'], y=daily_metrics['spend'],
                            mode='lines', name='Spend', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=daily_metrics['date'], y=daily_metrics['revenue'],
                            mode='lines', name='Revenue', line=dict(color='green')))
    fig.update_layout(title='Daily Spend vs Revenue', xaxis_title='Date', yaxis_title='Amount ($)',
                     hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Campaign comparison
    col1, col2 = st.columns(2)
    
    with col1:
        campaign_perf = df.groupby('campaign').agg({
            'spend': 'sum',
            'revenue': 'sum',
            'conversions': 'sum'
        }).reset_index()
        campaign_perf['roas'] = campaign_perf['revenue'] / campaign_perf['spend']
        
        fig = px.bar(campaign_perf, x='campaign', y='roas', 
                     title='ROAS by Campaign',
                     color='roas', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Conversion funnel
        funnel_data = pd.DataFrame({
            'Stage': ['Impressions', 'Clicks', 'Conversions'],
            'Count': [df['impressions'].sum(), df['clicks'].sum(), df['conversions'].sum()]
        })
        
        fig = px.funnel(funnel_data, x='Count', y='Stage', 
                       title='Conversion Funnel')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Channel Performance Analysis")
    
    # Channel metrics table
    channel_metrics = df.groupby('channel').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    channel_metrics['CTR'] = (channel_metrics['clicks'] / channel_metrics['impressions'] * 100).round(2)
    channel_metrics['CPA'] = (channel_metrics['spend'] / channel_metrics['conversions']).round(2)
    channel_metrics['ROAS'] = (channel_metrics['revenue'] / channel_metrics['spend']).round(2)
    
    st.subheader("Channel Metrics Summary")
    st.dataframe(channel_metrics.style.format({
        'spend': '${:,.0f}',
        'impressions': '{:,}',
        'clicks': '{:,}',
        'conversions': '{:,}',
        'revenue': '${:,.0f}',
        'CTR': '{:.2f}%',
        'CPA': '${:.2f}',
        'ROAS': '{:.2f}x'
    }), use_container_width=True)
    
    # Channel comparison visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(channel_metrics, values='spend', names='channel',
                     title='Spend Distribution by Channel')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(channel_metrics, x='CPA', y='ROAS', size='conversions',
                        color='channel', title='CPA vs ROAS by Channel',
                        hover_data=['spend', 'revenue'])
        fig.add_hline(y=1, line_dash="dash", line_color="gray",
                     annotation_text="Break-even ROAS")
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel trend analysis
    st.subheader("Channel Performance Over Time")
    
    channel_daily = df.groupby(['date', 'channel'])['roas'].mean().reset_index()
    fig = px.line(channel_daily, x='date', y='roas', color='channel',
                  title='ROAS Trend by Channel')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Customer Insights")
    
    # Customer segments (simulated)
    segments = ['New Customers', 'Returning Customers', 'VIP Customers', 'At-Risk']
    segment_data = pd.DataFrame({
        'Segment': segments,
        'Count': [2500, 1800, 500, 300],
        'Avg_LTV': [150, 450, 1200, 250],
        'Conversion_Rate': [2.5, 5.8, 12.3, 1.8]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(segment_data, x='Segment', y='Count',
                     title='Customer Distribution by Segment',
                     color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(segment_data, x='Conversion_Rate', y='Avg_LTV',
                        size='Count', color='Segment',
                        title='LTV vs Conversion Rate by Segment')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cohort analysis (simulated)
    st.subheader("Cohort Retention Analysis")
    
    # Generate cohort data
    cohorts = pd.date_range('2023-01-01', '2023-06-01', freq='M')
    retention_data = []
    
    for i, cohort in enumerate(cohorts):
        for month in range(6):
            retention_rate = 100 * (0.8 ** month) * np.random.uniform(0.9, 1.1)
            retention_data.append({
                'Cohort': cohort.strftime('%Y-%m'),
                'Month': month,
                'Retention': retention_rate
            })
    
    retention_df = pd.DataFrame(retention_data)
    retention_pivot = retention_df.pivot(index='Cohort', columns='Month', values='Retention')
    
    fig = px.imshow(retention_pivot, 
                    labels=dict(x="Months Since Acquisition", y="Cohort", color="Retention %"),
                    title="Customer Retention by Cohort",
                    color_continuous_scale='RdYlGn')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Predictive Analytics & Recommendations")
    
    # Forecast next month's performance
    st.subheader("Next Month Performance Forecast")
    
    # Simple forecast (in production, use proper time series models)
    last_30_days = df[df['date'] >= df['date'].max() - timedelta(days=30)]
    
    forecast_data = []
    for channel in df['channel'].unique():
        channel_data = last_30_days[last_30_days['channel'] == channel]
        avg_spend = channel_data['spend'].mean() * 30
        avg_conversions = channel_data['conversions'].mean() * 30
        avg_revenue = channel_data['revenue'].mean() * 30
        
        # Add some trend
        trend_factor = np.random.uniform(0.95, 1.15)
        
        forecast_data.append({
            'Channel': channel,
            'Predicted_Spend': avg_spend * trend_factor,
            'Predicted_Conversions': int(avg_conversions * trend_factor),
            'Predicted_Revenue': avg_revenue * trend_factor,
            'Predicted_ROAS': (avg_revenue * trend_factor) / (avg_spend * trend_factor)
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(forecast_df.style.format({
            'Predicted_Spend': '${:,.0f}',
            'Predicted_Conversions': '{:,}',
            'Predicted_Revenue': '${:,.0f}',
            'Predicted_ROAS': '{:.2f}x'
        }), use_container_width=True)
    
    with col2:
        fig = px.bar(forecast_df, x='Channel', y='Predicted_ROAS',
                     title='Predicted ROAS by Channel (Next Month)',
                     color='Predicted_ROAS', color_continuous_scale='Viridis')
        fig.add_hline(y=1, line_dash="dash", line_color="red",
                     annotation_text="Break-even")
        st.plotly_chart(fig, use_container_width=True)
    
    # Budget optimization recommendations
    st.subheader("Budget Optimization Recommendations")
    
    # Calculate optimal budget allocation
    total_budget = df['spend'].sum() / len(df['date'].unique()) * 30  # Monthly budget
    
    # Simple optimization based on ROAS
    channel_roas = df.groupby('channel')['roas'].mean().sort_values(ascending=False)
    
    recommendations = []
    for channel in channel_roas.index:
        current_spend = df[df['channel'] == channel]['spend'].sum() / len(df['date'].unique()) * 30
        current_share = current_spend / total_budget * 100
        
        # Recommend based on ROAS
        if channel_roas[channel] > 3:
            recommendation = "↑ Increase budget by 20%"
            color = "success"
        elif channel_roas[channel] > 2:
            recommendation = "→ Maintain current budget"
            color = "info"
        else:
            recommendation = "↓ Decrease budget by 15%"
            color = "warning"
        
        recommendations.append({
            'Channel': channel,
            'Current Budget %': current_share,
            'Current ROAS': channel_roas[channel],
            'Recommendation': recommendation
        })
    
    rec_df = pd.DataFrame(recommendations)
    
    for _, row in rec_df.iterrows():
        if "Increase" in row['Recommendation']:
            st.success(f"**{row['Channel']}**: {row['Recommendation']} (Current ROAS: {row['Current ROAS']:.2f}x)")
        elif "Maintain" in row['Recommendation']:
            st.info(f"**{row['Channel']}**: {row['Recommendation']} (Current ROAS: {row['Current ROAS']:.2f}x)")
        else:
            st.warning(f"**{row['Channel']}**: {row['Recommendation']} (Current ROAS: {row['Current ROAS']:.2f}x)")
    
    # A/B test results
    st.subheader("Recent A/B Test Results")
    
    ab_results = pd.DataFrame({
        'Test': ['Email Subject Line A/B', 'Landing Page CTA Color', 'Ad Copy Variation'],
        'Variant_A': [2.5, 3.2, 4.1],
        'Variant_B': [3.8, 3.1, 4.7],
        'Statistical_Significance': ['95%', '68%', '99%'],
        'Recommendation': ['Implement B', 'Continue Testing', 'Implement B']
    })
    
    st.dataframe(ab_results, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Dashboard updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.markdown("*Data refreshes every hour from integrated marketing platforms*")