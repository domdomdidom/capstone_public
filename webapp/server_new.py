from flask import Flask, render_template, request, jsonify 
import numpy as np
import pickle
import pandas as pd

cold_start_feature_df = pd.read_csv('cold_start_feature_df.csv', index_col = 'Customer ID')

def transform_one(one):

    one_dummy_Cgroup = pd.get_dummies(one['customer_group'])
    one_dummy_aff = pd.get_dummies(one['affiliation'])

    cgroups = list(np.unique(list(cold_start_feature_df['customer_group'].values)))
    affs = list(np.unique(list(cold_start_feature_df['affiliation'].values)))
    both = cgroups + affs

    both.remove(one['customer_group'].values.astype(str))
    both.remove(one['affiliation'].values.astype(str))

    for col in both:
        one[col] = 0

    one = one.join(one_dummy_Cgroup)
    one = one.join(one_dummy_aff)

    one['logged_subtotal'] = one['subtotal'].apply(lambda x: np.log(x) if x > 0 else 0)
    one['avg_price_item_logged'] = one['avg_price_item'].apply(lambda x: np.log(x) if x > 0 else 0)
    one['ship_logged'] = one['ship'].apply(lambda x: np.log(x) if x > 0 else 0)

    one = one.drop(columns=['customer_group', 'affiliation', 'subtotal', 'ship', 'avg_price_item'], axis=1)
    return one

# Create the app object that will route our calls
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template("home.html")



@app.route('/churning_man', methods = ['GET'])
def upload():

    return render_template("churning_man.html")


model = pickle.load(open('model1.pkl', 'rb'))    

@app.route('/inference', methods = ['POST'])
def inference():

    req = request.get_json()
    print(req)

    subtotal, ship, avg_price_item, subscriber_newsletter, uses_coupons, customer_group, affiliation  = req['subtotal'], req['ship'], req['avg_price_item'], req['subscriber_newsletter'], req['uses_coupons'], req['customer_group'], req['affiliation']

    data = {'subtotal':subtotal,  'ship':ship, 'avg_price_item':avg_price_item, 
            'subscriber_newsletter':subscriber_newsletter, 
            'uses_coupons':uses_coupons, 'customer_group':customer_group, 'affiliation':affiliation}

    index = [0]

    as_df= pd.DataFrame(data, index)        

    transformed = transform_one(as_df)

    prediction = list(model.predict(transformed))

    return jsonify({'avg_price_item':avg_price_item, 'subtotal':subtotal, 'ship':ship,  'customer_group':customer_group, 'uses_coupons':uses_coupons, 'affiliation':affiliation, 'subscriber_newsletter':subscriber_newsletter, 'prediction':np.round(prediction[0])})

#@app.route('/plot', methods = ["GET"])
#def plot():

#   df = pd.read_csv('cars.csv') # read in data
#    data = list(zip(df['mpg'], df['weight']))

#    return jsonify(data)




if __name__ == '__main__':
    model = pickle.load(open('model1.pkl', 'rb'))
    app.run(host='0.0.0.0', port=1212, debug=True)

    

