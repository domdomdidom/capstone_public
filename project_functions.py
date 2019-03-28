import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF as NMF_sklearn
import warnings; warnings.simplefilter('ignore')

custy_df = pd.read_csv('custy_data.csv', skiprows=None, index_col='Customer ID')
custy_df = custy_df.dropna(axis=0, how='all')
custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
order_df = pd.read_csv('order_data.csv', parse_dates=["Order Date"])
product_df = pd.read_csv('product_data.csv')
subscriber_df = pd.read_csv('subscribers-2019-03-27.csv')


def assemble_feature_df():
    
    feature_extraction_df = pd.DataFrame(columns=['AOV', 'Lapse', 'Order Freq', 'Total Ship', 'STD', 'AVGPERITEM', 'Is Sub'])
    feature_extraction_df = feature_extraction_df.reindex(custy_df.index)
    
    subscriber_emails = list(subscriber_df['Email'])

    for customer in custy_df.index.values:

        today = order_df["Order Date"].max() # this is the last order date we have in the order DF

        mask = order_df[order_df['Customer ID'] == customer] # mask for all orders under a customer
        if len(mask) == 0: # skip this customer if they've never ordered
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

            feature_extraction_df.loc[customer]['Order Freq'] = freq
            feature_extraction_df.loc[customer]['AOV'] = AOV
            feature_extraction_df.loc[customer]['Total Ship'] = total_ship
            feature_extraction_df.loc[customer]['Lapse'] = lapse
            feature_extraction_df.loc[customer]['STD'] = std_dev
            feature_extraction_df.loc[customer]['AVGPERITEM'] = avg_price_per_item
            feature_extraction_df.loc[customer]['Is Sub'] = is_subscriber


    feature_extraction_df['Customer Group'] = custy_df['Customer Group']
    feature_extraction_df['Affiliation'] = custy_df['Your Affiliation']
    
    return feature_extraction_df


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
    df['churn_bool'] = np.where(df['Lapse']>=days, 1, 0)

    
def log_cost_feats(df):
    
    df['AOV'] = df['AOV'].apply(lambda x: np.log(x) if x > 0 else 0)
    df['Total Ship'] = df['Total Ship'].apply(lambda x: np.log(x) if x > 0 else 0)
    
    return df

    
def binarize_cols(df):
    dummy_Cgroup = pd.get_dummies(df['Customer Group'])
    dummy_aff = pd.get_dummies(df['Affiliation'])
    
    dummy_merged = dummy_aff.merge(dummy_Cgroup, on='Customer ID')
    returned = df.join(dummy_merged)
    returned2 = returned.drop(columns=['Affiliation', 'Customer Group'], axis=1)
    
    return returned2
    
def make_TT_split(df):
    y = df['churn_bool']
    X = df.drop(columns=['Lapse', 'churn_bool'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    return X_train, X_test, y_train, y_test