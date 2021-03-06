# Using Machine Learning to Predict Churn and Analyze Customer Lifetime
![](images/RockTape-Logo-R-B-RGB.png) 

# Table of Contents
1. [Introduction](#Introduction)
2. [Packages Used](#Packages-Used)
3. [Workflow](#Workflow)
4. [Additional Functions](#Additional-Functions)
5. [Discussion of Results](#Discussion-of-Results)
6. [Criticisms and Future Work](#Criticisms-and-Future-Work)

# Introduction
Being able to identify users who are at risk of churning is quite important - we can segment our customer base & pivot our marketing tactics to certain users, or spend resources to improve areas where a business is weak. BigCommerce tracks lots of stats for sales, customers and 3rd party marketing plugins. In this repo, I'll explore some of these data and see if we can gain some useful insights on identifying potential churn!

Take a look at [my super-cool webapp!](http://52.90.122.192:1212/churning_man) I pay good money for Amazon to host this bad boy! Please excuse the Internet 1.0 CSS. The webapp vectorizes all the inputs into an array, runs the array through the model.predict() function, and returns the output.

# Packages Used
    Sklearn
    RFpimp
    Pandas
    Numpy
    Flask
    Pickle

# Workflow

  0. First things first - we need to get the raw data from BigCommerce into a usable form. Export all your customers from BigCommerce, as well as your order data. These exports will probably take a while, depending on how large your company is. Now, export your entire product catalog. Finally, head over to your 3rd party marketing tab and export your newsletter subscribers. All four exports should be in a CSV.
  
  You can load in your CSVs with the ReadMyFiles class, if you'd like. The four functions in here will clean up the exports. The default BigCommerce exports are a little ugly - we can drop some columns right off the bat, parse our dates and skip every other line in the customer file. 
  
  It's important to note that your customer and subscriber exports will only reflect the most up-to-date information for your customers. We don't have access to historical data, unfortunately, so we make the assumption that there has not been any historical changes that will afffect our results (such as a customer's 'customer_group' changing, or unsubscribing from our newsletter). We can access the historical order data, so no worries there! 
  
 ![](images/mydata.png)
  
 
  1. Lets take a look at our first class, InitialExtraction. There are two functions in here: assemble_feature_dfs and make_historical_purchase_matrix. Here's an overview:
  
    assemble_feature_dfs : Loops through each customer in your customer dataframe and aggregates statistics about him/her in      two dataframes: one for historical statistics and one for our "cold start" problem. 
    
    Here are the features engineered for our standard feature dataframe.
    
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
    
    A real business use of this idea is to ultimately be able to predict who is at risk of leaving when we have *very little       information about them*, such as in the case of a brand new customer. To assemble our cold start dataframe, we are only       going to consider a customer's very first order for most of our feature engineering. We've tossed any columns that may         potentially leak information.
    
    'avg_order_value' - the subtotal of a customer's first purchase
    'avg_price_item' - subtotal / number of items purchased
    'subscriber_newsletter' - does the customer subscribe to our newsletter (binary variable)
    'uses_coupons' - did the customer use a coupon for their first order (does not include cart-level discounts, binary           variable)
    
    'time_as_customer' - this is our target. time lapsed since a customer joined (from custy_df) and their most recent order        (output in days). We consider all orders a customer has placed when we make this.
  
    
    make_historical_purchase_matrix : Loops through each customer and records their itemized purchase 
    in an mxn sparse matrix where m is number of customers, and n is number of products in your product library. 
    
    Ex)          widget    thingamabob      gizmo
          Larry     6           3             0
          Claire    0           0             1
          Martin    1           0             0
    
Specify your order and customer dataframes in the constructor. You'll also need to specify your product and subscriber dataframes in the following functions. Be sure you save these outputs as variables so we can access them later! These three functions take a considerable amount of time to run (for 500,000 orders and 70,000 customers, it takes about 25 minutes) - you may want to consider saving the outputs as CSVs if you're going to be working with them often.  

    init_extract = InitialExtraction(order_df, custy_df)

    feature_df, cold_start_feature_df = init_extract.assemble_feature_dfs(subscriber_df)
    historical_purchase_df, historical_purchase_matrix = init_extract.make_historical_purchase_matrix(product_df)
    
Here's our feature dataframe:
![](images/features.png)
                  
  2. Yay! Hopefully our feature extractions didn't take too long to compile. Let's move onto our next class, Transform. 
  In the constructor, specify which feature_df you'd like to use (either the vanilla feature_df, which I am hereby going to just call feature_df, or your cold_start_feature_df). The workflows for both are slightly different, I'm going to go over the feature_df first. 
      
First, we need to make a binary variable for our target, churn. I define churn as not having placed an order in the past 365 days. You can specify your definition in the function:

      mytransform = Transform(feature_df)
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
      coldtransform.transform_cold_start_data()
      
The last function in here is do_NMF. The NMF class is not applicable to our cold_start problem since we are working with such limited data. NMF, or non-negative matrix factorization, links our products to our customers via latent "topics", or clusters of weighted similarity. 

I made the executive decision to include this function within the Transform class. We are effectively treating the results of NMF as engineered features. Go ahead and save this version as our final transformation. If you'd like to see the 25 top products associated with your latent NMF topics, set get_top_products = True!

        transformed = mytransform.do_NMF(historical_purchase_matrix, product_df, get_top_products=False)
    
![](images/NMF.png)

After we concat the original features with the NMF weights, here's what out data look like:

![](images/features_plut_NMF.png)

3. Cool, now that our data is transformed, we can split it up. Within the Splitter class, we have two functions: split_for_churn and split_for_cold_start. Let's just worry about churn for now, but know that you can split for cold_start by just switching the functions. In the constructor, specify your transformed, cleaned-up dataframe. 

        makesplits = Splitter(transformed)
        X_train, X_test, y_train, y_test = makesplits.split_for_churn(transformed)
      
      
4. At this point, we have our original engineered feature matrix combined with our weights matrix. We're ready to model now! We're going to put this dataframe full of super-awesome features inside of a random forest. Go ahead and call the Model class.

        mymodel = Model()

We first want to fit our model to our training set:

        mymodel.fit(X_train, y_train)
        
Next, let's do a predict on our test set:    
    
        mymodel.predict(X_test)
 
Finally, we're going to score the results of our test set against our predictions:

        mymodel.score(X_test, y_test, sklearn=True)
 
The score function will output accuracy, precision and recall scores. It also calculates the feature importances. The top feature importances are those features that did the best job of "unmixing" the labels or predicting the target. They provided the most significant reduction in gini impurity per split. You'll see a graph of the top 15 feature importances for your model here. 

Permutation importances show the other side of feature importance - the P.I. score decreases when a feature is not available to the model. It's a good way of validating our feature importances. The top features hilighted by our permutation importance graph and our feature importance graph should be mostly the same. I used RFpimp to validate the Sklearn feature importances. Here they are back-to-back. They look pretty similar, but you can see slight differences!

![](images/feat_imp_vs_pimp.png) 


6. Let us revisit our old friend, cold_start. 
As a reminder, we didn't do any NMF for this problem. We just have the transformed cold_start_df. Let's do a TT split on this bad boy. Make sure you're using the right dataframe :)

       makesplits = Splitter()
       X_train, X_test, y_train, y_test= makesplits.split_for_cold_start(transformed)
       
As a reminder, our data look like this:

![](images/cold_start.png) 

       
Word! Ok, time to model. Just like the instructions above, we're going to fit, predict, and score our model EXCEPT FOR TWO BIG DIFFERENCES:

        1. We are using a REGRESSOR instead of a CLASSIFIER. Our target, time_as_customer, is a continuous variable. 
        It didn't seem appropriate to use historical data to model a future event ('will this customer churn?'), 
        so we're switching up the question to 'what is the lifespan of this customer?'. 
        It's a variation of the same question, just framed in a way that makes a bit more sense 
        with the data we're using. 
        
        2. We are using a GradientBoost instead of a RandomForest! Since we didn't do NMF, and we lost about half 
        of our original features, a more robust model seems to perform better here. Note that both GB and RF are 
        ensemble decision tree methods. 
        
I digress. Time to model.
        
        coldModel = Model_Cold_Start()
        
        coldModel.fit(X_train, y_train)
        coldModel.predict(X_test)
        coldModel.score(X_test, y_test)
        
Feature Importances are good, but not great. Sometimes the "unmixing" can be a result of random chance. Feature Importances are also relitave to the amount of columns we have. If we have two columns that encode similar information, the feature importances will be artifically lower because of the sheer number of columns, even though this is good information. Our partial dependency plots show us exactly how our outcome changes with that particular variable. For example, being a chiropractor has a positive impact on lifetime. Interestingly enough, free shipping on a first order doesn't seem to have much of an effect on lifetime. 

![](images/pdplots.png) 


How'd we do this time?!?! Since we aren't scoring a classifier here, we don't have accuracy, precision and recall (those are methods of scoring true negatives, false positives, etc). We evaluate our model with Root Mean Squared Error. Our baseline_mse is just the root mean squared error of the [mean of our y_train] * len(y_test). Our cold_start model looks to be about 15-20% better than our baseline! Yay improvement! 


# Additional Functions

In my package, there is a function called get_items_associated. This function takes in your historical purchase matrix, and a trimmed feature dataframe narrowed down to a specific customer faction you'd like to examine. It uses linear algebra (NMF) to compile a list of the top products for this faction. 

You can toggle the hyperparameters of this function, but the defaults are as follows. For example, if you'd like to examine only Chiropractors who have churned, but were customers for at least 2 years prior, you would do:

      sliced_df = feature_df[ (feature_df['Medical - Chiropractor'] == 1) & 
                              (feature_df['day_since_last_order] > 365) &
                               (feature_df['time_as_customer']) ]
      
      get_items_associated(historical_purchase_matrix, 
                          sliced_df, product_df, n_topics=5, max_iters=350, 
                          n_churniest_topics=3, n_churniest_items=25)
      
      
Our results:

      [('RockBand', 3), ('2" Digital Camouflage * - DISCONTINUED- DO NOT USE', 3), 
      ('KneeCaps - Knee Support and Protection', 3), ('2" Black Skull *', 3), 
      ('2" Black Logo', 3), ('2" Pink Camouflage', 2), ('2" Pink Logo (discontinued as of 11/11/2016 df)', 2), 
      ('2" Purple', 2), ('2" H2O Black - extra sticky', 2), ('Talons - Hand Protection', 2), 
      ('4" H2O Mini Big Daddy Black Logo', 2), ('Assassins Knee Sleeves - Manifesto', 2), ('Coffee Cup', 2), 
      ('2" Bulk H2O Black Logo - extra sticky', 2), ('RockSauce - Skin Prep & Pain* Reliever *', 2), 
      ('Power Taping Posters', 2), ("Product Brochure * (DO NOT USE - see 'brochure' for retail/medical options df 3.13.17)", 2), ('Rehab Poster *', 2), ('2" H2O Black Logo - extra sticky', 2), 
      ('RockTape Brochure', 2), ('Power Sample Strips Black', 2), 
      ('pHast Legs 90ct - The Paleo Supplement - discontinued 06/03/2014', 2), 
      ('2" Yellow * DISCONTINUED PRODUCT', 2), ('2" Green Camouflage', 2), ('2" Orange *', 2)]
      
These seem to be largely discontinued products (an asterisk also denotes discontinutiy), which we may expect from looking at people who have churned. I would interpret these as our "weakest products" - ones that actively contribute to customer dissatisfaction or fall short of expectations in some way. 

You can slice your dataframe in an infinite number of configurations to grab useful product stats!
      
# Discussion of Results

  Using the vanilla feature_df, I was able to correctly classify a customer as churned/not churned about 80% of the time (relitave to a baseline of about a 50/50 split, equivalent to a random guess). We used NMF to extract 5 latent features, and wrapped all our features with a random forest classifier. We were able to identify certain features that weighed more heavily on a customer's liklihood to churn.

    It looks like getting free samples and testers doesn't help keep our customers loyal
    Customer type is more important that a customer's order history
    Our students churn at a higher rate than most
  
  This is all well and good, but the real business use case of this project is forecasting a new customer's lifetime, where we have limited information about them. We had access to way less features for this task and we weren't able to use NMF to identify those latent features. Using a GradientBoostRegressor, we were able to improve predicting new customers lifetimes by 15%. Take a look at the Partial Dependency Plots above to see how the top 6 features affect a customer's lifetime!
  
    New chiropractors are more likely to stick around longer
    Using a coupon with a first order positively correlates with lifespan
    Buying expensive items contributes negatively to lifespan
    Free or reduced shipping doesn't seem to extend lifetime
    
# Criticisms and Future Work

  RockTape stopped dividing up their affiliations with such fine granularity in 2015. There may be some unavoidable information leakage here, since people who are assigned to "antiquated" affiliations are by default, older customers. 
  
  Create a confusion matrix for promos like free shipping & discount use. We're losing money by offering free shipping with no benefit to us.
