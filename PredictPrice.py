#import libraries
from __future__ import division
from datetime import datetime, timedelta,date
import pandas as pd
import os
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from sklearn.cluster import KMeans

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import plotly for visualization
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

os.system('cls')
df = pd.read_excel('Data_KHTT_profile.xlsx',converters={'NgayCapNhatCuoi':pd.to_datetime}, index_col=0) 

tx_user = pd.DataFrame(df['MaThe7'].unique())
tx_user.columns = ['MaThe7']


# calculate recency
#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = df.groupby('MaThe7').NgayCapNhatCuoi.max().reset_index()
tx_max_purchase.columns = ['MaThe7','MaxPurchaseDate']


#we take our observation point as the max invoice date in our dataset
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['MaThe7','Recency']], on='MaThe7')

#print(tx_user.Recency.describe())

tx_user.Recency.describe().to_excel('D:\\Recency.xlsx','Sheet1')


#plot a recency histogram
plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency - Tính chất mới xảy ra'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig)

#Cleaning data frame
df_fillNAN = tx_user.fillna(tx_user.min())
"""
sse={}
tx_recency = df_after[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()
"""

#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_fillNAN[['Recency']])
df_fillNAN['RecencyCluster'] = kmeans.predict(df_fillNAN[['Recency']])



df_result = order_cluster('RecencyCluster', 'Recency',df_fillNAN,False)


df_result.to_excel('D:\\Recency.xlsx','Sheet2')
print(df_result)