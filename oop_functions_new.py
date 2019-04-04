import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF as NMF_sklearn
import warnings; warnings.simplefilter('ignore')
import rfpimp
from sklearn.metrics import confusion_matrix, precision_score, recall_score, log_loss, roc_auc_score, accuracy_score
from collections import Counter
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

custy_df = pd.read_csv('custy_data.csv', skiprows=None, index_col='Customer ID', parse_dates=["Date Joined"]).dropna(axis=0, how='all')
custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
order_df = pd.read_csv('order_data.csv', parse_dates=["Order Date"])
product_df = pd.read_csv('product_data.csv')
subscriber_df = pd.read_csv('subscribers-2019-03-27.csv')


class InitialExtraction():
    ''' Perform initial extractions on the raw data. This class creates an itemized purchase matrix 
    (n_cols = len(products), n_rows = len(customers)) and a feature extraction dataframe.
     
     Parameters
     ----------
    
     
     Attributes
     ----------
     
     '''

    def __init__(self, order_df, custy_df):

        self.order_df = order_df
        self.custy_df = custy_df
        self.feature_extraction_df = None

        self.historical_purchase_matrix = None
        self.historical_purchase_df = None

    def assemble_feature_df(self, subscriber_df):
    
        feature_extraction_df = pd.DataFrame(columns=['avg_order_value', 'days_since_last_order', 'time_as_customer', 
                                                    'order_freq', 'n_total_orders', 'ship_total', 'std_order_value', 
                                                    'avg_price_item', 'subscriber_newsletter', 'uses_coupons'])

        feature_extraction_df = feature_extraction_df.reindex(self.custy_df.index)
        subscriber_emails = list(subscriber_df['Email'])

        for customer in self.custy_df.index.values:

            today = self.order_df["Order Date"].max() # this is the last order date we have in the order DF

            mask = self.order_df[self.order_df['Customer ID'] == customer] # mask for all orders under a customer
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
                if num_orders == 1: 
                    std_dev = 0 
                    
                else: std_dev = round(mask['Subtotal'].std(), 3)

                # total $ spent on shipping
                total_ship = round(mask['Shipping Cost'].sum(), 3)

                # average price per items purchased
                avg_price_per_item = mask['Subtotal'].sum()/mask['Total Quantity'].sum()

                # is subscriber
                if mask['Customer Email'].values[0] in subscriber_emails: 
                    is_subscriber = 1

                else: is_subscriber = 0

                # uses coupons
                coupons_list = [x for x in list(mask['Coupon Details']) if str(x) != 'nan']
                if len(coupons_list) > 0:
                    uses_coupons = 1
                
                else: uses_coupons = 0

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
                feature_extraction_df.loc[customer]['uses_coupons'] = uses_coupons


        feature_extraction_df['customer_group'] = self.custy_df['Customer Group']
        feature_extraction_df['affiliation'] = self.custy_df['Your Affiliation']
        
        print("initial feature extraction completed.")
        self.feature_extraction_df = feature_extraction_df.dropna(thresh=3)
        return self.feature_extraction_df


    def make_historical_purchase_matrix(self, product_df):
    
        historical_purchase_df = pd.DataFrame(0, index=self.feature_extraction_df.index, columns=product_df['Product ID'])

        for customer in self.feature_extraction_df.index.values:

            mask = self.order_df[self.order_df['Customer ID'] == customer] # mask for all orders under a customer

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
        print("historical itemized matrix assembled.") 

        self.historical_purchase_df = historical_purchase_df
        self.historical_purchase_matrix = historical_purchase_df.as_matrix()    

        return self.historical_purchase_df, self.historical_purchase_matrix

class Transform():
    ''' Transform initial feature dataframe into a binarized dataframe with all the
    price features logged and a churn boolean added
     
     Parameters
     ----------
    feature_df = the results of 'assemble_feature_matrix', should contain 1 column for 'Affiliation' and 1 column for 'Customer Group' 
    (these two are the only string cols, the rest are numeric)
     
     Attributes
     ----------
     this class uses one hot encoding, logs cost features, makes a churn boolean, and trims the dataframe
     option to fit an initial random forest to these features
     '''

    def __init__(self, feature_df):

        self.feature_df = feature_df

    def make_churn(self, days):
        self.feature_df['churn_bool'] = np.where(self.feature_df['days_since_last_order'] >= days, 1, 0)

    def binarize(self):
        
        dummy_Cgroup = pd.get_dummies(self.feature_df['customer_group'])
        dummy_aff = pd.get_dummies(self.feature_df['affiliation'])
    
        dummy_merged = dummy_aff.merge(dummy_Cgroup, on='Customer ID')
        self.feature_df = self.feature_df.merge(dummy_merged, on='Customer ID')

    def log_cost_features(self):

        self.feature_df['avg_order_value_logged'] = self.feature_df['avg_order_value'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['ship_total_logged'] = self.feature_df['ship_total'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['avg_price_item_logged'] = self.feature_df['avg_price_item'].apply(lambda x: np.log(x) if x > 0 else 0)

    def drop_cols_and_nan_rows(self):
        
        self.feature_df = self.feature_df.drop(columns=['ship_total', 'avg_order_value', 'customer_group', 'affiliation', 'days_since_last_order', 'avg_price_item'], axis=1)
        self.feature_df.dropna(thresh=4, inplace=True)
        self.feature_df = self.feature_df.fillna(0)
        
        return self.feature_df

    def scale_data(self):

        scaler = StandardScaler()
        self.feature_df = scaler.fit_transform(self.feature_df)
        
        return self.feature_df
    
    def do_NMF(self, historical_purchase_matrix, product_df, n_topics, max_iters):
        
        nmf = NMF_sklearn(n_components=n_topics, max_iter=max_iters, alpha=0.0)
        W = nmf.fit_transform(historical_purchase_matrix) # how much each customer belongs to each "topic"
        H = nmf.components_ # how much each item belongs to each "topic"
        
        W_df = pd.DataFrame(W) # weights matrix only
        self.feature_df = pd.concat([self.feature_df.reset_index(drop=False), W_df.reset_index(drop=True)], axis=1) # combine weights matrix with feature_df
        return self.feature_df.set_index('Customer ID')

class Splitter():

    def __init__(self):

        self.feature_df = None

    def split_for_churn(self, feature_df):

        X = feature_df.drop(columns=['churn_bool'], axis=1)
        y = feature_df['churn_bool']
            
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        return X_train, X_test, y_train, y_test

    def split_for_cold_start(self, feature_df):

        ''' What happens when we have a brand new customer?
        We have: id, CGroup, Affiliation, AOV, is_subscriber, avg_price_item at minimum
        We can do NMF and identify that customer with groups/products more likely to churn'''

        cold_start = feature_df[feature_df['n_total_orders'] == 1 ]

        X = cold_start.drop(columns=['time_as_customer', 'order_freq', 'n_total_orders', 'std_order_value', 'churn_bool'], axis=1)
        y = cold_start['time_as_customer']
                
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        return X_train, X_test, y_train, y_test


class Model():
    ''' Use sklearn's implementation of NMF combined with some supervised learning techniques.
     
     Parameters
     ----------

     
     Attributes
     ----------
     '''
    def __init__(self):
        
        self.X = None
        self.y = None
        self.model = None
        self.y_pred = None

    def fit(self, X, y):
        
        self.X = X
        self.y = y
        
        model = RandomForestClassifier(max_depth=10, n_estimators=200)
        self.model = model.fit(self.X, self.y)
        print('Model fitted.')

    def predict(self, X):

        self.y_pred = self.model.predict(X)
        return self.y_pred

    def score(self, X_test, y_test):

        print("The following are the results of a random forest fit to the original features appended to the Weights Matrix:")
        print("\naccuracy:", round(self.model.score(X_test, y_test),3))
        print("precision:", round(precision_score(y_test, self.y_pred),3))
        print("recall:", round(recall_score(y_test, self.y_pred),3))
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature ranking:")
        for feat in range(0,15):
            print("%d. %s (%f)" % (feat + 1, self.X.columns[indices[feat]], importances[indices[feat]]))
            
        # plot it
        plt.figure(figsize=(12,6))
        plt.ylabel('Feature Name', size=12)
        plt.xlabel('Relative Feature Importance', size=12)
        plt.title('Feature Importances', size=18)
        feat_importances = pd.Series(importances, index=self.X.columns)
        feat_importances.nlargest(15).plot(kind='barh')
        plt.grid(color='grey', ls=':')
        plt.show()
        


if __name__ == '__main__':

    ### INITIAL EXTRACTION
    init_extract = InitialExtraction(order_df, custy_df)

    feature_df = init_extract.assemble_feature_df(subscriber_df)
    historical_purchase_df, historical_purchase_matrix = init_extract.make_historical_purchase_matrix(product_df)

    ### TRANSFORM DATA
    mytransform = Transform(feature_df)

    mytransform.make_churn(365)
    mytransform.binarize()
    mytransform.log_cost_features()
    mytransform.drop_cols_and_nan_rows()
    transformed = mytransform.do_NMF(historical_purchase_matrix, product_df, n_topics=5, max_iters=350)

    ### SPLITTER
    makesplits = Splitter()

    churn_train_X, churn_test_X, churn_train_y, churn_test_y = makesplits.split_for_churn(transformed)

    ### MODEL DATA
    mymodel = Model()

    mymodel.fit(churn_train_X, churn_train_y)
    mymodel.predict(churn_test_X)
    mymodel.score(churn_test_X, churn_test_y)





