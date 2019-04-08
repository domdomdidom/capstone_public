import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF as NMF_sklearn
import warnings; warnings.simplefilter('ignore')
import rfpimp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, precision_score, recall_score, log_loss, roc_auc_score, accuracy_score, mean_squared_error
from collections import Counter
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import partial_dependence

class ReadMyFiles():
    ''' These will format BigCommerce output files specifically :) '''

    def __init__(self):

        self.custy_df = None
        self.order_df = None
        self.subscriber_df = None
        self.product_df = None

    def read_customer(self, filepath):

        custy_df = pd.read_csv(filepath, skiprows=None, index_col='Customer ID', parse_dates=["Date Joined"]).dropna(axis=0, how='all')
        self.custy_df = custy_df.drop(columns=['Rep Name', 'Education'], axis=1)
        return self.custy_df
    
    def read_order(self, filepath):

        self.order_df = pd.read_csv(filepath, parse_dates=["Order Date"])
        return self.order_df

    def read_product(self, filepath):

        self.product_df = pd.read_csv(filepath)
        return self.product_df

    def read_marketing(self, filepath):

        self.subscriber_df= pd.read_csv(filepath)
        return self.subscriber_df


class InitialExtraction():
    ''' Perform initial extractions on the raw data. This class creates an itemized purchase matrix 
    (n_cols = len(products), n_rows = len(customers)), a feature extraction dataframe, and a cold start feature
    extraction dataframe.
     
     Parameters
     ----------
    order_df
    custy_df
    subscriber_df (or list)
     
     Attributes
     ----------
     
     '''

    def __init__(self, order_df, custy_df):

        self.order_df = order_df
        self.custy_df = custy_df
        self.feature_extraction_df = None
        self.cold_start_feature_extraction_df = None

        self.historical_purchase_matrix = None
        self.historical_purchase_df = None

    def assemble_feature_dfs(self, subscriber_df):
    
        feature_extraction_df = pd.DataFrame(columns=['avg_order_value', 'days_since_last_order', 'time_as_customer', 
                                                    'order_freq', 'n_total_orders', 'avg_ship', 'std_order_value', 
                                                    'avg_price_item', 'subscriber_newsletter', 'uses_coupons'])

        cold_start_feature_extraction_df = pd.DataFrame(columns=['avg_order_value', 'time_as_customer', 'ship',
                                                    'avg_price_item', 'subscriber_newsletter', 'uses_coupons'])

        cold_start_feature_extraction_df = cold_start_feature_extraction_df.reindex(self.custy_df.index)
        feature_extraction_df = feature_extraction_df.reindex(self.custy_df.index)

        subscriber_emails = subscriber_df['Email'].astype(list)

        for customer in self.custy_df.index.values:

            today = self.order_df["Order Date"].max() # this is the last order date we have in the order DF

            mask = self.order_df[self.order_df['Customer ID'] == customer] # mask for all orders under a customer
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
                
                # time as customer
                time_as_customer = (mask["Order Date"].max() - self.custy_df['Date Joined'][customer]).days

                # average order value
                AOV = round(mask['Subtotal'].sum() / len(mask), 3)
                if num_orders == 1: 
                    std_dev = 0
                else: std_dev = round(mask['Subtotal'].std(), 3)

                # avg $ spent on shipping
                avg_ship = round(mask['Shipping Cost'].sum() / len(mask), 3)

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
                
                feature_extraction_df.loc[customer]['avg_order_value'] = AOV
                feature_extraction_df.loc[customer]['std_order_value'] = std_dev
                feature_extraction_df.loc[customer]['n_total_orders'] = num_orders
                feature_extraction_df.loc[customer]['days_since_last_order'] = lapse
                feature_extraction_df.loc[customer]['time_as_customer'] = time_as_customer
                feature_extraction_df.loc[customer]['order_freq'] = freq
                feature_extraction_df.loc[customer]['avg_ship'] = avg_ship
                feature_extraction_df.loc[customer]['avg_price_item'] = avg_price_per_item
                feature_extraction_df.loc[customer]['subscriber_newsletter'] = is_subscriber
                feature_extraction_df.loc[customer]['uses_coupons'] = uses_coupons    

                ### GET COLD START
                ''' What happens when we have a brand new customer?
                We only have 'avg_order_value', 'avg_price_item', 'subscriber_newsletter', 'uses_coupons', 'customer_group' and 'affiliation' 
                I've truncated everybody's orders after their first order -- can we predict how long they will 'be customers'? time_as_customer is defined
                as time between date_joined and last order date (true) '''

                mask_zero = mask.head(1)

                # average order value
                AOV = mask_zero['Subtotal'].sum()

                # avg $ spent on shipping
                ship = round(mask_zero['Shipping Cost'].sum(), 3)

                # average price per items purchased
                avg_price_per_item = mask_zero['Subtotal'].sum()/mask_zero['Total Quantity'].sum()

                # is subscriber
                if mask_zero['Customer Email'].values[0] in subscriber_emails: 
                    is_subscriber = 1
                else: is_subscriber = 0

                # uses coupons
                coupons_list = [x for x in list(mask_zero['Coupon Details']) if str(x) != 'nan']
                if len(coupons_list) > 0:
                    uses_coupons = 1
                else: uses_coupons = 0

                cold_start_feature_extraction_df.loc[customer]['avg_order_value'] = AOV
                cold_start_feature_extraction_df.loc[customer]['time_as_customer'] = time_as_customer
                cold_start_feature_extraction_df.loc[customer]['avg_price_item'] = avg_price_per_item
                cold_start_feature_extraction_df.loc[customer]['subscriber_newsletter'] = is_subscriber
                cold_start_feature_extraction_df.loc[customer]['uses_coupons'] = uses_coupons
                cold_start_feature_extraction_df.loc[customer]['ship'] = ship


        dfs = [feature_extraction_df, cold_start_feature_extraction_df]
        for df in dfs:
            df['customer_group'] = self.custy_df['Customer Group']
            df['affiliation'] = self.custy_df['Your Affiliation']
            df.dropna(thresh=3, inplace=True)

        print("initial feature extractions completed for vanilla and cold start.")

        self.feature_extraction_df = feature_extraction_df
        self.cold_start_feature_extraction_df = cold_start_feature_extraction_df

        return feature_extraction_df, cold_start_feature_extraction_df

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
        self.feature_df['avg_ship_logged'] = self.feature_df['avg_ship'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['avg_price_item_logged'] = self.feature_df['avg_price_item'].apply(lambda x: np.log(x) if x > 0 else 0)

    def drop_old_cols(self):
        
        self.feature_df = self.feature_df.drop(columns=['avg_ship', 'avg_order_value', 'customer_group', 'affiliation', 'days_since_last_order', 'avg_price_item'], axis=1)
        self.feature_df.dropna(thresh=4, inplace=True)
        self.feature_df = self.feature_df.fillna(0)
        
        return self.feature_df

    def transform_cold_start_data(self):

        self.feature_df['avg_order_value_logged'] = self.feature_df['avg_order_value'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['avg_price_item_logged'] = self.feature_df['avg_price_item'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df['ship_logged'] = self.feature_df['ship'].apply(lambda x: np.log(x) if x > 0 else 0)
        self.feature_df = self.feature_df.drop(columns=['avg_order_value', 'ship', 'customer_group', 'affiliation', 'avg_price_item'], axis=1)
        
        return self.feature_df
    
    def do_NMF(self, historical_purchase_matrix, product_df, get_top_products=False):
        
        nmf = NMF_sklearn(n_components=5, max_iter=450)
        W = nmf.fit_transform(historical_purchase_matrix) # how much each customer belongs to each "topic"
        H = nmf.components_ # how much each item belongs to each "topic"
        
        W_df = pd.DataFrame(W) # weights matrix only
    
        merged_df = pd.concat([self.feature_df.reset_index(drop=False), W_df.reset_index(drop=True)], axis=1) # combine weights matrix with feature_df
        merged_df = merged_df.rename(columns={0:'Consumer Rehab/Single Rolls', 1:'Education/Movement Professionals', 2:'Consumer Fitness', 3:'Marketing & Promo', 4:'CrossFit'})
        merged_df = merged_df.set_index('Customer ID')
        self.feature_df = merged_df
        
        if get_top_products == True:
            
            print("Here are the top products for %s topics" % (5))
            for topic in range(0, 5):
                indicies = H[topic].argsort()[-25:]
                print("\n")
                print(product_df['Name'][indicies])
            
        return self.feature_df

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

        y = feature_df['time_as_customer']
        X = feature_df.drop(columns=['time_as_customer'], axis=1)

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

    def score(self, X_test, y_test, use_rfpimp=True, use_sklearn=True):

        print("The following are the results of a random forest fit to the original features appended to the Weights Matrix:")
        print("\naccuracy:", round(self.model.score(X_test, y_test),3))
        print("precision:", round(precision_score(y_test, self.y_pred),3))
        print("recall:", round(recall_score(y_test, self.y_pred),3))

        pimp_imps = rfpimp.importances(self.model, self.X, self.y)
        rfpimp.plot_importances(pimp_imps[0:9], yrot=0,
                                label_fontsize=12,
                                width=12,
                                minheight=1.5,
                                vscale=2.0,
                                imp_range=(0, pimp_imps['Importance'].max() + .03),
                                color='#484c51',
                                bgcolor='#F1F8FE',  # seaborn uses '#F1F8FE'
                                xtick_precision=2,
                                title='Permutation Importances')

        if use_sklearn == True:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nFeature ranking:")
            for feat in range(0,10):
                print("%d. %s (%f)" % (feat + 1, self.X.columns[indices[feat]], importances[indices[feat]]))

            # plot feat imps
            plt.figure(figsize=(12,6))
            plt.ylabel('Feature Name', size=12)
            plt.xlabel('Relative Feature Importance', size=12)
            plt.title('Sklearn Feature Importances', size=18)
            feat_importances = pd.Series(importances, index=self.X.columns)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.grid(color='grey', ls=':')
            plt.show()

class Model_Cold_Start():
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
        self.baseline = None

    def fit(self, X, y):
        
        self.X = X
        self.y = y

        self.baseline = self.y.mean()
        
        model = GradientBoostingRegressor(n_estimators = 150, learning_rate = 0.05, 
                                   max_features = 'sqrt', max_depth = 10)
        self.model = model.fit(self.X, self.y)
        print('Model fitted.')

    def predict(self, X):

        self.y_pred = self.model.predict(X)
        return self.y_pred

    def score(self, X_test, y_test):

        self.baseline = [self.baseline] * len(y_test)

        baseline_MSE = np.sqrt(mean_squared_error(y_test, self.baseline))
        print('Baseline Root Mean Squared Error:', round(baseline_MSE , 0))
        
        model_MSE = np.sqrt(mean_squared_error(y_test, self.y_pred))
        print('Model Root Mean Squared Error:', round(model_MSE, 0))

        print('Reduction in RMSE: %', ((model_MSE - baseline_MSE) / baseline_MSE))
        
        ### PD PLOT
        importances = self.model.feature_importances_
        sorted_imps = sorted(importances)[::-1]
        indicies = np.argsort(importances)[::-1]
        names = self.X.columns[indicies]
        N_COLS = 3

        pd_plots = [partial_dependence(self.model, target_feature, X=self.X, grid_resolution=50)
                    for target_feature in indicies]
        pd_plots = list(zip(([pdp[0][0] for pdp in pd_plots]), ([pdp[1][0] for pdp in pd_plots])))

        fig, axes = plt.subplots(nrows=3, ncols=N_COLS, sharey=True, 
                                figsize=(12.0, 8.0))

        for i, (y_axis, x_axis) in enumerate(pd_plots[0:(3*N_COLS)]):
            ax = axes[i//N_COLS, i%N_COLS]
            ax.plot(x_axis, y_axis, color="purple")
            ax.set_xlim([np.min(x_axis), np.max(x_axis)])
            text_x_pos = np.min(x_axis) + 0.05*(np.max(x_axis) - np.min(x_axis))
            ax.text(text_x_pos, 7.5,
                    "Feature Importance " + str(round(sorted_imps[i],2)), 
                    fontsize=12, alpha=0.7)
            ax.set_xlabel(names[i])
            ax.grid()
            
        plt.suptitle("Partial Dependence Plots (Ordered by Feature Importance)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        

### ADDITIONAL FUNCTIONS ###         
def get_items_associated(historical_purchase_matrix, feature_df, product_df, n_topics=5, max_iters=350, n_churniest_topics=3, n_churniest_items=25):
    ''' 
    Get products most associated with a particular group
    Parameters
    ----------
    index of a dataframe slice (pandas.core.indexes.numeric.Int64Index)

        
    Attributes
    ----------  

    '''
    slice = feature_df.index.astype(int)

    nmf = NMF_sklearn(n_components=n_topics, max_iter=max_iters, alpha=0.0)
    W = nmf.fit_transform(historical_purchase_matrix) # how much each customer belongs to each "topic"
    H = nmf.components_ # how much each item belongs to each "topic"
    
    sums = W[slice].sum(axis=0)
    churniest_topics = sums.argsort()[-n_churniest_topics:] 
    
    c = Counter()
    
    for topic in churniest_topics:
        indicies = H[topic].argsort()[-50:]
    
        for product in product_df['Name'][indicies]:
            c[product] += 1
            
    return c.most_common(n_churniest_items)

if __name__ == '__main__':

    ### LOAD MY FILES
    readme = ReadMyFiles()
    custy_df = readme.read_customer('custy_data.csv')
    order_df = readme.read_order('order_data.csv')
    product_df = readme.read_product('product_data.csv')
    subscriber_df = readme.read_marketing('subscribers-2019-03-27.csv')
    print('files read.')

    ### INITIAL EXTRACTION
    init_extract = InitialExtraction(order_df, custy_df)

    feature_df, cold_start_feature_df = init_extract.assemble_feature_dfs(subscriber_df)

    historical_purchase_df, historical_purchase_matrix = init_extract.make_historical_purchase_matrix(product_df)

    ### TRANSFORM DATA
    mytransform = Transform(feature_df)

    mytransform.make_churn(365)
    mytransform.binarize()
    mytransform.log_cost_features()
    mytransform.drop_old_cols()
    transformed = mytransform.do_NMF(historical_purchase_matrix, product_df, get_top_products=False)

    ### SPLITTER
    makesplits = Splitter()

    X_train, X_test, y_train, y_test = makesplits.split_for_churn(transformed)

    ### MODEL DATA
    mymodel = Model()

    mymodel.fit(X_train, y_train)
    mymodel.predict(X_test)
    mymodel.score(X_test, y_test, use_sklearn=True)


    ### EXPLORE COLD START

    ### TRANSFORM DATA
    coldtransform = Transform(cold_start_feature_df)
    coldtransform.binarize()
    transformed = coldtransform.transform_cold_start_data()

    ### SPLITTER
    makesplits = Splitter()
    X_train, X_test, y_train, y_test= makesplits.split_for_cold_start(transformed)

    ### MODEL DATA
    coldModel = Model_Cold_Start()
    coldModel.fit(X_train, y_train)
    coldModel.predict(X_test)
    coldModel.score(X_test, y_test)
    








