import pandas as pd

# data can be downloaded from the url: https://www.kaggle.com/vikrishnan/boston-house-prices 
#df = pd.read_csv('./06_input_data.csv')
df = pd.read_csv('D:/ticket/patients__payment_csv.csv')

# Understanding data
print (df.shape)
print (df.columns)
print(df.head(5))
print(df.info())
print(df.describe())
print(df.groupby('appointment_token').size())

# Dropping null value columns which cross the threshold
a = df.isnull().sum()
print (a)
b =  a[a>(0.05*len(a))]
print (b)
df = df.drop(b.index, axis=1)
print (df.shape)

# Replacing null value columns (text) with most used value
a1 = df.select_dtypes(include=['object']).isnull().sum()
print (a1)
print (a1.index)
for i in a1.index:
	b1 = df[i].value_counts().index.tolist()
	print (b1)
	df[i] = df[i].fillna(b1[0])
	
# Replacing null value columns (int, float) with most used value
a2 = df.select_dtypes(include=['integer','float']).isnull().sum()
print (a2)
b2 = a2[a2!=0].index 
print (b2)
#df = df.fillna(df[b2].mode().to_dict(orient='records')[0])
modes = df[b2].mode().to_dict(orient='records')
if modes:
    df = df.fillna(modes[0])

# Creating new columns from existing columns
# print (df.shape)
# a3 = df['YrSold'] - df['YearBuilt']
# b3 = df['YrSold'] - df['YearRemodAdd']
# df['Years Before Sale'] = a3
# df['Years Since Remod'] = b3
# print (df.shape)

# Dropping unwanted columns
# df = df.drop(["id","transaction_date","payment_category"], axis=1) 
# print (df.shape)

# Dropping columns which has correlation with target less than threshold
# target='total_amount'
# x = df.select_dtypes(include=['integer','float']).corr()[target].abs()
# print (x)  
# df=df.drop(x[x<0.4].index, axis=1)
# print (df.shape)

# Checking for the necessary features after dropping some columns
l1 = ["token","appointment_token","doctor_details_token ","patients_token","due_amount","paid_amount","refund_amount","total_amount"]
l2 = []
for i in l1:
    if i in df.columns:
        l2.append(i)
       
print("testing",l2)


# Getting rid of nominal columns with too many unique values
# for i in l2:
#     len(df[i].unique())>10
#     df=df.drop(i, axis=1)
# print (df.columns)
print("completed")

df.to_csv('D:/ticket/preprocessing_data.csv', index=False)
