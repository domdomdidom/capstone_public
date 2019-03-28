import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF as NMF_sklearn
import warnings; warnings.simplefilter('ignore')
import rfpimp
from sklearn.metrics import confusion_matrix, precision_score, recall_score

custy_df = pd.read_csv('custy_data.csv', skiprows=None, index_col='Customer ID', parse_dates=["Date Joined"])
custy_df = custy_df.dropna(axis=0, how='all')
custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
order_df = pd.read_csv('order_data.csv', parse_dates=["Order Date"])
product_df = pd.read_csv('product_data.csv')
subscriber_df = pd.read_csv('subscribers-2019-03-27.csv')


def assemble_feature_df():
    
    feature_extraction_df = pd.DataFrame(columns=['avg_order_value', 'days_since_last_order', 'time_as_customer', 'order_freq', 'n_total_orders', 'ship_total', 'std_order_value', 'avg_price_item', 'subscriber_newsletter'])
    feature_extraction_df = feature_extraction_df.reindex(custy_df.index)
    
    subscriber_emails = list(subscriber_df['Email'])

    for customer in custy_df.index.values:

        today = order_df["Order Date"].max() # this is the last order date we have in the order DF

        mask = order_df[order_df['Customer ID'] == customer] # mask for all orders under a customer
        if len(mask) <= 0: # skip this customer if they've never ordered
            pass

        else:
            # get freq
            delta = (mask["Order Date"].max() - mask["Order Date"].min()).days / 365 # time spanned between first and last order
            num_orders = len(mask['Order Date'].values)

            if delta == 0: # fix for div by zero
                freq = 1
            else: freq = round((num_orders/delta), 3) 

            # days since last order
            lapse = (today - mask["Order Date"].max()).days
            
            # time as customer
            time_as_customer = (mask["Order Date"].max() - custy_df['Date Joined'][customer]).days

            # average order value
            AOV = round(mask['Subtotal'].sum() / len(mask), 3)
            std_dev = round(mask['Subtotal'].std(), 3)

            # total $ spent on shipping
            total_ship = round(mask['Shipping Cost'].sum(), 3)

            # average price per items purchased
            avg_price_per_item = mask['Subtotal'].sum()/mask['Total Quantity'].sum()

            # is subscriber
            if mask['Customer Email'].values[0] in subscriber_emails: 
                is_subscriber = 1

            else: is_subscriber = 0

            # throw all this info into a matrix indexed by customer id number. i can probably throw this in a for loop to clean it up a bit
            
            feature_extraction_df.loc[customer]['avg_order_value'] = AOV
            feature_extraction_df.loc[customer]['std_order_value'] = std_dev
            feature_extraction_df.loc[customer]['n_total_orders'] = num_orders
            feature_extraction_df.loc[customer]['days_since_last_order'] = lapse
            feature_extraction_df.loc[customer]['time_as_customer'] = time_as_customer
            feature_extraction_df.loc[customer]['order_freq'] = freq
            feature_extraction_df.loc[customer]['ship_total'] = total_ship
            feature_extraction_df.loc[customer]['avg_price_item'] = avg_price_per_item
            feature_extraction_df.loc[customer]['subscriber_newsletter'] = is_subscriber

    feature_extraction_df['customer_group'] = custy_df['Customer Group']
    feature_extraction_df['affiliation'] = custy_df['Your Affiliation']
    
    return feature_extraction_df.dropna(thresh=3)


def make_historical_purchase_matrix():
    
    historical_purchase_df = pd.DataFrame(0, index=custy_df.index, columns=product_df['Product ID'])

    for customer in custy_df.index.values:

        mask = order_df[order_df['Customer ID'] == customer] # mask for all orders under a customer

        for order in mask['Product Details'].values:
            itemized = order.split('|') # split each "itemized order line"

            for line in itemized:
                keep, rubbish = line.split(', Product SKU') # get rid of everything after prodct SKU

                prod_id, prod_qty = keep.split(',')

                rubbish, prod_id = prod_id.split(':') # 
                rubbish, prod_qty  = prod_qty.split(':')

                prod_id = int(prod_id.strip()) # strip whitespace
                prod_qty = int(prod_qty.strip())

                if prod_id not in list(product_df['Product ID']):
                    pass
                
                else: historical_purchase_df[prod_id][customer] += prod_qty
            
    historical_purchase_matrix = historical_purchase_df.as_matrix()
    return historical_purchase_matrix

def do_NMF(X, n_topics, max_iters, n_items):
    
    nmf = NMF_sklearn(n_components=n_topics, max_iter=max_iters, alpha=0.0)
    W = nmf.fit_transform(X) # how much each customer belongs to each "topic"
    H = nmf.components_ # how much each item belongs to each "topic"
    
    for topic in range(0, n_topics-1):
        indicies = H[topic].argsort()[-n_items:-1]
        print("\n")
        print(product_df['Name'][indicies])


def make_churn(df, days):
    df['churn_bool'] = np.where(df['days_since_last_order']>=days, 1, 0)

    
def log_cost_features(df):
    
    df['avg_order_value_logged'] = df['avg_order_value'].apply(lambda x: np.log(x) if x > 0 else 0)
    df['ship_total_logged'] = df['ship_total'].apply(lambda x: np.log(x) if x > 0 else 0)

    df = df.drop(columns=['ship_total', 'avg_order_value'], axis=1)
    
    return df.fillna(0)

def binarize_cols(df):
    dummy_Cgroup = pd.get_dummies(df['customer_group'])
    dummy_aff = pd.get_dummies(df['affiliation'])
    
    dummy_merged = dummy_aff.merge(dummy_Cgroup, on='Customer ID')
    returned = df.join(dummy_merged)
    returned2 = returned.drop(columns=['affiliation', 'customer_group'], axis=1)
    
    return returned2
    
def make_TT_split(df):

    y = df['churn_bool']
    X = df.drop(columns=['days_since_last_order', 'churn_bool'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    return X_train, X_test, y_train, y_test

def score_rf(df, n_feature_importances):
    
    X_train, X_test, y_train, y_test = make_TT_split(df)
    
    rf = RandomForestClassifier(oob_score=True)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    imp = rfpimp.importances(rf, X_test, y_test)
    cols_to_drop = list(imp[imp['Importance'] <= 0].index)
    
    if len(cols_to_drop) > 0:
        df = df.drop(columns=cols_to_drop, axis=1) # drop the unimportant columns, run the model again
        rf = RandomForestClassifier(oob_score=True)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        imp = rfpimp.importances(rf, X_test, y_test)
        cols_to_drop = list(imp[imp['Importance'] <= 0].index)
    
    print(confusion_matrix(y_test, y_pred))
    print("\naccuracy:", rf.score(X_test, y_test))
    print("precision:", precision_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))
    print('\n')
    print(imp[0:n_feature_importances])
