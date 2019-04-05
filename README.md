# Predicting Churn and Analyzing Customer Lifetime with a Combination of User, Sales and Marketing Data

Being able to identify users who are at risk of churning is quite important - we can segment our customer base & pivot our marketing tactics to certain users, or spend resources to improve areas where a business is weak. BigCommerce tracks lots of stats for sales, customers and 3rd party marketing plugins. In this repo, I'll explore some of these data and see if we can gain some useful insights on identifying potential churn!

Workflow:
  0. First things first - we need to get the raw data from BigCommerce into a usable form. Export all your customers from BigCommerce, as well as your order data. These exports will probably take a while, depending on how large your company is. Now, export your entire product catalog. Finally, head over to your 3rd party marketing tab and export your newsletter subscribers. All four exports should be in a CSV.
  
  It's important to note that your customer export and subscriber export will only reflect the most up-to-date information for your customers. We don't have access to historical data, unfortunately, so we make the assumption that there has not been any historical changes that will afffect our results (such as a customer's 'customer_group' changing, or unsubscribing from our newsletter). We can access the historical order data, so no worries there! 
  
  1. Lets take a look at our first class, InitialExtraction, that lives in InitialExtraction.py. There are three functions in here: assemble_feature_df, make_historical_purchase_matrix and assemble_cold_start_feature_df. Here's an overview:
  
    assemble_feature_df : Loops through each customer in your customer dataframe, does initial calculations and records them in a new dataframe. Skips customers who have never ordered. This function aggregates the following statistics:
    
    'avg_order_value' - the mean price of all orders this customer has placed
    'days_since_last_order' - time lapsed between customer's most recent order and "today" (the date of last in order_df)
    'time_as_customer' - time lapsed since a customer joined (from custy_df) and their most recent order (output in days)
    'order_freq' - number of orders / time between first and last order (output in days)
    'n_total_orders' - number of orders customer has placed
    'ship_total' - how much the customer has spent on shipping in their lifetime
    'std_order_value' - standard deviation of all orders customer has placed
    'avg_price_item' - average order value / number of items purchased
    'subscriber_newsletter' - does the customer subscribe to our newsletter (binary variable)
    'uses_coupons' - has the customer used at least one coupon (does not include cart-level discounts, binary variable)
    
    
    make_historical_purchase_matrix : Loops through each customer and records their itemized purchase in an mxn sparse matrix where m is number of customers, and n is number of products in your product library. 
    
    Ex)          widget    thingamabob      gizmo
          Larry     6           3             0
          Claire    0           0             1
          Martin    1           0             0
    
    
    assemble_cold_start_feature_df : This is all fine and dandy so far, but a real business use of this idea is to ultimately be able to predict who is at risk of leaving when we have *very little information about them*, such as in the case of a brand new customer. To assemble our cold start dataframe, we are only going to consider a customer's very first order for most of our feature engineering. 
    
    'avg_order_value' - the subtotal of a customer's first purchase
    'avg_price_item' - subtotal / number of items purchased
    'subscriber_newsletter' - does the customer subscribe to our newsletter (binary variable)
    'uses_coupons' - did the customer use a coupon for their first order (does not include cart-level discounts, binary variable)
    
    'time_as_customer' - this is our target. time lapsed since a customer joined (from custy_df) and their most recent order (output in days). we consider all orders when we make this.
    
Specify your order and customer dataframes in the constructor. You'll also need to specify your product and subscriber dataframes in the following functions. Be sure you save these outputs as variables so we can access them later! These three functions take a considerable amount of time to run (for 500,000 orders and 70,000 customers, it takes about 25 minutes) - you may want to consider saving the outputs as CSVs if you're going to be working with them often.  

    init_extract = InitialExtraction(order_df, custy_df)

    feature_df = init_extract.assemble_feature_df(subscriber_df)
    historical_purchase_df, historical_purchase_matrix = init_extract.make_historical_purchase_matrix(product_df)
    
    cold_start_feature_df = init_extract.assemble_cold_start_feature_df(subscriber_df)
                                                   
  2. Yay! Hopefully our feature extractions didn't take too long to compile. Let's move onto our next class, Transform. 
  In the constructor, specify which feature_df you'd like to use (either the vanilla feature_df, which I am hereby going to just call feature_df, or your cold_start_feature_df). The workflows for both are slightly different, I'm going to go over the feature_df first. 
  
      mytransform = Transform(feature_df)
      
First, we need to make a binary variable for our target, churn. I define churn as not having placed an order in the past 365 days. You can specify your definition in the function:

      mytransform.make_churn(365)

Now let's binarize our Affiliation and Customer Group columns:
      
      mytransform.binarize()
      
The next step is to log all our price features:

      mytransform.log_cost_features()
      
Finally, we do a little cleanup to drop all the columns we just transformed, as well as ones who might leak infomation about churn: 

      transformed = mytransform.drop_old_cols()
  
The workflow for cold_start is mostly the same, except some of the transformations are rolled together in a single function, transform_cold_start_data. Follow these steps to transform your cold_start dataframe:

      coldtransform = Transform(cold_start_feature_df)
      coldtransform.binarize()
      transformed = coldtransform.transform_cold_start_data()
      
3. Cool, now that our data is transformed, we can split it up. Within the Splitter class, we have two functions: split_for_churn and split_for_cold_start. Let's just worry about churn for now, but know that you can split for cold_start by just switching the functions. In the constructor, specify your transformed, cleaned-up dataframe. 

        makesplits = Splitter(transformed)
        X_train, X_test, y_train, y_test = makesplits.split_for_churn(transformed)
      
4. Nice! We're ready to move onto the NMF class. The NMF class is not applicable to our cold_start problem since we are working with such limited data. NMF, or non-negative matrix factorization, links our products to our customers via latent "topics", or clusters of weighted similarity. 
 
5. At this point, we have our original engineered feature matrix combined with our weights matrix. We're ready to model now! We're going to put this dataframe full of super-awesome features inside of a random forest. Go ahead and call the Model class.

        mymodel = Model()

We first want to fit our model to our training set:

        mymodel.fit(X_train, y_train)
        
Next, let's do a predict on our test set:    
    
        mymodel.predict(X_test)
 
Finally, we're going to score the results of our test set against our predictions:

        mymodel.score(X_test, y_test)
 
The score function will output accuracy, precision and recall scores. It also calculates the feature importances. The top feature importances are those features that give us the best information gain per split, they are our "good predictors". You'll see a graph of the top 15 feature importances for your model here. How'd we do?!

6. Let us revisit our old friend, cold_start. 
As a reminder, we didn't do any NMF for this problem. We just have the transformed cold_start_df. Let's do a TT split on this bad boy. Make sure you're using the right dataframe :)

       makesplits = Splitter()
       X_train, X_test, y_train, y_test= makesplits.split_for_cold_start(transformed)
       
Word! Ok, time to model. Just like the instructions above, we're going to fit, predict, and score our model EXCEPT FOR TWO BIG DIFFERENCES:
        1. We are using a REGRESSOR instead of a CLASSIFIER. Our target, time_as_customer, is a continuous variable. It didn't seem appropriate to use historical data to model a future event ('is this customer going to churn'), so we're switching up the question to 'how long was this person a customer'. It's a variation of the same question, just framed in a way that makes a bit more sense with the data we're using. 
        2. We are using a GradientBoost instead of a RandomForest! Since we didn't do NMF, and we lost about half of our original features, a more robust model seems to perform better here. Note that both GB and RF are ensemble decision tree methods. 
        
        coldModel = Model_Cold_Start()
        
        coldModel.fit(X_train, y_train)
        coldModel.predict(X_test)
        coldModel.score(X_test, y_test)
        
How'd we do this time?!?! Since we aren't scoring a classifier here, we don't have accuracy, precision and recall (those are methods of scoring true negatives, false positives, etc). We evaluate our model with Mean Squared Error. Our baseline_mse is just the mean_squared_error of the [mean of our y_train] * len(y_test). Our cold_start model looks to be about 33% better than our baseline! Yay improvement! 
      
      
