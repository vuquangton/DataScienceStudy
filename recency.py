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
from order_cluster import order_cluster
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
df = pd.read_excel('Data_Sales_SKU_CUST.xlsx',converters={'INVOICEDATE':pd.to_datetime}, index_col=0) 

tx_user = pd.DataFrame(df['CUSTNUM'].unique())
tx_user.columns = ['CUSTNUM']

tx_max_purchase = df.groupby('CUSTNUM').INVOICEDATE.max().reset_index()
tx_max_purchase.columns = ['CUSTNUM','MaxPurchaseDate']

tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

tx_user = pd.merge(tx_user, tx_max_purchase[['CUSTNUM','Recency']], on='CUSTNUM')

print(tx_user['Recency'].describe())

#plot a recency histogram
plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency - Hot-Warm-Cold'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig)



sse={}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])
df_result = order_cluster('RecencyCluster', 'Recency',tx_user,False)

tx_usr = pd.DataFrame(df['CUSTNUM'].unique())
tx_usr.columns = ['CUSTNUM']

tx_frequency = df.groupby('CUSTNUM').INVOICEDATE.count().reset_index()
tx_frequency.columns = ['CUSTNUM','Frequency']

tx_usr = pd.merge(tx_usr, tx_frequency, on='CUSTNUM')

#plot the histogram
plot_data = [
    go.Histogram(
        x=tx_usr.query('Frequency < 1000')['Frequency']
        # x=tx_usr['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig)

#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_usr[['Frequency']])
tx_usr['FrequencyCluster'] = kmeans.predict(tx_usr[['Frequency']])

#order the frequency cluster
tx_usr = order_cluster('FrequencyCluster', 'Frequency',tx_usr,True)

#see details of each cluster
print(tx_usr.groupby('FrequencyCluster')['Frequency'].describe())
print(tx_user.groupby('RecencyCluster')['Recency'].describe())


tx_usr2 = pd.DataFrame(df['CUSTNUM'].unique())
tx_usr2.columns = ['CUSTNUM']

tx_revenue = df.groupby('CUSTNUM').VALUE.sum().reset_index()
tx_revenue.columns = ['CUSTNUM','Revenue']
tx_usr2 = pd.merge(tx_usr2,tx_revenue, on='CUSTNUM')


plot_data = [
    go.Histogram(
        x=tx_usr2.query('Revenue < 10000')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig)

#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_usr2[['Revenue']])
tx_usr2['RevenueCluster'] = kmeans.predict(tx_usr2[['Revenue']])


#order the cluster numbers
tx_usr2 = order_cluster('RevenueCluster', 'Revenue',tx_usr2,True)

#show details of the dataframe
print(tx_usr2.groupby('RevenueCluster')['Revenue'].describe())

tx_user['Overallscore'] =  tx_user['RecencyCluster'] + tx_usr['FrequencyCluster'] + tx_usr2['RevenueCluster']

tx_user.groupby('Overallscore')['Recency','Frequency','Revenue'].mean()
print(tx_user.groupby('Overallscore')['Recency','Frequency','Revenue'].mean())