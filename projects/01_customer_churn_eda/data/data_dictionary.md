# Customer Churn Dataset - Data Dictionary

## Overview
This synthetic dataset represents customer information from a telecommunications company, designed to analyze factors contributing to customer churn.

## Variables

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| customer_id | Integer | Unique customer identifier | 1-1000 |
| tenure_months | Integer | Number of months customer has been with company | 1-72 |
| monthly_charges | Float | Average monthly bill amount | $20-$150 |
| total_charges | Float | Total amount charged to customer | Varies |
| contract_type | Categorical | Type of customer contract | Month-to-month, One year, Two year |
| payment_method | Categorical | Customer's payment method | Electronic check, Mailed check, Bank transfer, Credit card |
| internet_service | Categorical | Type of internet service | DSL, Fiber optic, No |
| num_services | Integer | Number of additional services subscribed | 1-8 |
| tech_support | Categorical | Whether customer has tech support | Yes, No |
| churn | Binary | Whether customer churned (1) or not (0) | 0, 1 |

## Data Quality Notes
- No missing values in synthetic dataset
- All monetary values in USD
- Churn rate approximately 26.5% to reflect realistic industry standards