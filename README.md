# Predicting Churn with a Combination of User, Sales and Marketing Data

Being able to identify users who are at risk of churning is quite important - we can segment our customer base & pivot our marketing tactics to certain users, or spend resources to improve areas where a business is weak. BigCommerce tracks lots of stats for sales, customers and 3rd party plugins. In this repo, I'll explore some of these data and see if we can gain some useful insights on identifying potential churn!

Workflow:
  1. Extract features from order and customer data. Establish a new dataframe (hereby referred to as feature_extraction_df). This dataframe contains the following columns (as of 3/28): 
  'avg_order_value', 'time_since_last_order', 'order_frequency', 'total_ship_spend', 'std_of_aov', 'avg_price_per_item', 'is_newsletter_subsciber'
  
    Functions used: assemble_feature_df
  
  2. Create our binary target column for churn. I'm defining churn as: 
    - has ordered at least twice prior
    - is not a "guest" checkout
    - has not ordered in the past 365 days
    
    Functions used: make_churn
    
  3. Perform addition transformations our feature dataframe. I logged any price data (average_order_value and ship_total), dropped rows with nans that exceeded my threshold of 3, and binarized my 'customer_group' and 'affiliation' fields!
  
    Functions used: log_cost_features, binarize_cols
  
  4. Random Forest Time! Let's score our model with a confusion matrix, recall, precision and accuracy scores. We also get our top feature importances.
  
    Functions used: make_TT_split, score_rf
  
  5. Use NMF to identify latent topics and groupings of customers. 
  
    Functions used: make_historical_purchase_matrix, do_NMF
