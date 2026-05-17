# Part 1 Dataset: Customer Churn Neural Network Dataset

## File
`customer_churn_nn.csv`

## Goal
Build a neural network model to predict whether a customer is likely to churn.

## Target Column
- `churn`: 1 = customer churned, 0 = customer retained

## Feature Notes
- Categorical columns: `region`, `plan_type`, `contract_type`, `payment_method`
- Numerical columns: tenure, charges, login days, tickets, delays, data usage, satisfaction, complaint recency, discounts, referrals
- `customer_id` is an identifier and should not be used as a predictive feature.

## Suggested Student Tasks
- Explore class distribution
- Encode categorical features
- Scale numerical features
- Train/test split
- Build feed-forward neural network
- Compare hyperparameter configurations
