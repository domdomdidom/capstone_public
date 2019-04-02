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

custy_df = pd.read_csv('custy_data.csv', skiprows=None, index_col='Customer ID', parse_dates=["Date Joined"]).dropna(axis=0, how='all')
custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
order_df = pd.read_csv('order_data.csv', parse_dates=["Order Date"])
product_df = pd.read_csv('product_data.csv')
subscriber_df = pd.read_csv('subscribers-2019-03-27.csv')


def assemble_feature_df(custy_df, order_df, subscriber_df):
    
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
    
    print("initial feature extraction completed.")
    return feature_extraction_df.dropna(thresh=3)

def make_historical_purchase_matrix(trimmed_df, order_df, product_df):
    
    historical_purchase_df = pd.DataFrame(0, index=trimmed_df.index, columns=product_df['Product ID'])

    for customer in trimmed_df.index.values:

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
    print("historical itemized matrix assembled.")        
    return historical_purchase_df, historical_purchase_df.as_matrix()

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
        
    def drop_cols_and_nan_rows(self):
        
        self.feature_df = self.feature_df.drop(columns=['ship_total', 'avg_order_value', 'customer_group', 'affiliation', 'days_since_last_order'], axis=1)
        self.feature_df.dropna(thresh=4, inplace=True)
        self.feature_df = self.feature_df.fillna(0)
        
        return self.feature_df
    
    def do_initial_rf(self, n_feature_importances=15):
    
        X = self.feature_df.drop(columns=['churn_bool'], axis=1)
        y = self.feature_df['churn_bool']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    
        model = RandomForestClassifier(oob_score=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print("The following are the results of a random forest fit to the original feature extractions only:")
        print("\naccuracy:", round(model.score(X_test, y_test),3))
        print("precision:", round(precision_score(y_test, y_pred),3))
        print("recall:", round(recall_score(y_test, y_pred),3))
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature ranking:")
        for feat in range(0,15):
            print("%d. %s (%f)" % (feat + 1, X_train.columns[indices[feat]], importances[indices[feat]]))
            
            
class NMF():
    ''' Use sklearn's implementation of NMF combined with some supervised learning techniques.
     
     Parameters
     ----------

     
     Attributes
     ----------
     '''

    def __init__(self):

        self.historical_purchase_matrix = None

        self.H = None
        self.W = None
        
        self.n_topics = None

        self.W_train = None
        self.W_test = None
        self.y_train = None
        self.y_test = None

    def do_NMF(self, historical_purchase_matrix, n_topics, max_iters):
        
        self.historical_purchase_matrix = historical_purchase_matrix
        self.n_topics = n_topics
        
        nmf = NMF_sklearn(n_components=n_topics, max_iter=max_iters, alpha=0.0)
        self.W = nmf.fit_transform(self.historical_purchase_matrix) # how much each customer belongs to each "topic"
        self.H = nmf.components_ # how much each item belongs to each "topic"
    
        return self.W, self.H
    
    def top_products_for_topics(self, product_df, n_items):
        print("Here are the top products for %s topics" % (self.n_topics))
        for topic in range(0, self.n_topics):
            indicies = self.H[topic].argsort()[-n_items:]
            print("\n")
            print(product_df['Name'][indicies])

    def do_logreg(self, feature_df):

        self.W_train, self.W_test, self.y_train, self.y_test = train_test_split(self.W, feature_df['churn_bool'], random_state=10)

        model = LogisticRegression(solver='lbfgs')

        self.model = model
        self.model.fit(self.W_train, self.y_train)

        y_pred = (self.model.predict_proba(self.W_test)[:,1] >= .66) # may wanna just make this a .5?

        print("The following are the results of a logistic regression done only using the Weights Matrix from NMF:")
        print("\ncoef weights:", self.model.coef_)
        print("accuracy:", round(accuracy_score(self.y_test, y_pred),3))
        print("precision:", round(precision_score(self.y_test, y_pred),3))
        print("recall:", round(recall_score(self.y_test, y_pred),3))

    def do_RF(self, feature_df):

        W_df = pd.DataFrame(self.W)
        churn_bool = feature_df['churn_bool']
        feature_df = feature_df.drop(columns=['churn_bool'], axis=1)

        merged_df = pd.concat([feature_df.reset_index(drop=True), W_df.reset_index(drop=False)], axis=1)
        new_index = feature_df.index[0:len(merged_df)]
        merged_df = merged_df.drop(columns=['index'], axis=1)
        merged_df.reindex(new_index).dropna()
        
        self.W_train, self.W_test, self.y_train, self.y_test = train_test_split(merged_df, churn_bool, random_state=10)

        model = RandomForestClassifier(oob_score=True)

        self.model = model
        self.model.fit(self.W_train, self.y_train)

        y_pred = self.model.predict(self.W_test)

        print("The following are the results of a random forest fit to the original features appended to the Weights Matrix:")
        print("\naccuracy:", round(self.model.score(self.W_test, self.y_test),3))
        print("precision:", round(precision_score(self.y_test, y_pred),3))
        print("recall:", round(recall_score(self.y_test, y_pred),3))
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature ranking:")
        for feat in range(0,15):
            print("%d. %s (%f)" % (feat + 1, self.W_train.columns[indices[feat]], importances[indices[feat]]))

        
    def get_churniest_items(self, feature_df, product_df, n_topics, max_iters=250, n_churniest_topics=3, n_churniest_items=25):
        
        self.n_topics = n_topics

        churned_index = np.array((feature_df[(feature_df['churn_bool'] == 1) & (feature_df['time_as_customer'] > 365)].index).astype(int))
        churning_mask_matrix = self.historical_purchase_matrix[churned_index]
        
        W, H = self.do_NMF(churning_mask_matrix, n_topics=self.n_topics, max_iters=max_iters)

        sums = W.sum(axis=0)
        churniest_topics = sums.argsort()[-n_churniest_topics:] 
    
        c = Counter()
    
        for topic in churniest_topics:
            indicies = H[topic].argsort()[-50:]
            for product in product_df['Name'][indicies]:
                c[product] += 1

        return c.most_common(n_churniest_items)


if __name__ == '__main__':
    #step 1 - extract initial features
    feature_df = assemble_feature_df(custy_df, order_df, subscriber_df)

    #step 2 - transform data (binarize cols, log cost feats, make a churn bool)
    churnTransformer = Transform(feature_df)

    churnTransformer.make_churn(365)
    churnTransformer.binarize()
    churnTransformer.log_cost_features()
    transformed = churnTransformer.drop_cols_and_nan_rows()
    churnTransformer.do_initial_rf(n_feature_importances=15)
    print("Transformations completed.")

    #step 3 - assemble itemized historical purchase matrix
    historical_purchase_df, historical_purchase_matrix = make_historical_purchase_matrix(feature_df, order_df, product_df)

    #step 4 - do NMF and get top products for each topic
    myNMF = NMF()

    W_0, H_0 = myNMF.do_NMF(historical_purchase_matrix, n_topics=5, max_iters=50)
    myNMF.top_products_for_topics(product_df, n_items=25)
    myNMF.do_logreg(transformed)
    myNMF.do_RF(transformed)
    myNMF.get_churniest_items(transformed, product_df, n_topics=5, n_churniest_topics=3, n_churniest_items=25)


