import os 
import json
import pandas as pd
import numpy
#from sklearn.externals import joblib
import joblib


#s = pd.read_json('D:/ticket/inputdata.json')
s = pd.read_json('D:/ticket/inputdata.json')
print(s)

p = joblib.load("D:/ticket/07_output_salepricemodel.pkl")

#print("testing",p)
#print(p.feature_names_in_)


r = p.predict(s)

print (str(r))