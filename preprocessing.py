#step 1 import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import dataset from local 
data = pd.read_csv("Dataset.csv")
data.fillna(data.mean(),inplace = True)
# data cleaning
X = data.iloc[:,:3].values
Y = data.iloc[:,-1].values
#filling missing values

#Concept of Dummy Variable, Handling the conflict of them
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[:,0] = label_encoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
 # Training and testing data (divide data into two parts)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size = 0.2,random_state = 0)
#normalize the data
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X_test  = SS.fit_transform(X_test)
X_train = SS.fit_transform(X_train)
