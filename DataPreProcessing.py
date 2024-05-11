import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset.csv",low_memory = False)

new_df= df.drop(['step','nameOrig','nameDest','isFlaggedFraud','oldbalanceDest','newbalanceDest'], axis = 1)

new_df['type']=new_df['type'].map({'PAYMENT':1, 'TRANSFER':4, 'CASH_OUT':2, 'DEBIT':5, 'CASH_IN':3})

X = new_df.iloc[:,:-1].values
y = new_df.iloc[:,-1].values

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)