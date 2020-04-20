# import libraries
from __future__ import division
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist
from order_cluster import order_cluster
from sklearn.cluster import KMeans

df_source = pd.read_excel('Data_Sales_SKU_CUST.xlsx')
tx_user = pd.DataFrame(df_source['CUSTNUM'].unique())
tx_user.columns = ['CUSTNUM']

tx_max_purchase = df_source.groupby('CUSTNUM').INVOICEDATE.max().reset_index()
tx_max_purchase.columns = ['CUSTNUM','MaxPurchaseDate']

tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['CUSTNUM','Recency']], on='CUSTNUM')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

#order the cluster numbers
tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,True)

#get order counts for each user and create a dataframe with it
tx_frequency = df_source.groupby('CUSTNUM').INVOICEDATE.count().reset_index()
tx_frequency.columns = ['CUSTNUM','Frequency']
tx_user = pd.merge(tx_user, tx_frequency, on='CUSTNUM')
#add this data to our main dataframe


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

#order the cluster numbers
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)


tx_revenue = df_source.groupby('CUSTNUM').VALUE.sum().reset_index()
tx_revenue.columns = ['CUSTNUM','Monetary']

tx_user = pd.merge(tx_user, tx_revenue, on='CUSTNUM')

#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Monetary']])
tx_user['MonetaryCluster'] = kmeans.predict(tx_user[['Monetary']])


#order the cluster numbers
tx_user = order_cluster('MonetaryCluster', 'Monetary',tx_user,True)

#calculate overall score and use mean() to see details
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['MonetaryCluster']
tx_user.groupby('OverallScore')['Recency','Frequency','Monetary'].mean()

tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 



# ax = tx_user['OverallScore'].value_counts().plot(kind='bar', figsize=(15, 5), fontsize=12)
# ax.set_xlabel("RFM Score", fontsize=12)
# ax.set_ylabel("Count", fontsize=12)
# x =tx_user.query("Segment=='Low-Value'").reset_index()
# y =tx_user.query("Segment=='Mid-Value'").reset_index()
# plt.scatter(x, y, c='red')
# y = tx_user.query("Segment=Mid-Value'").count()
# plt.scatter(x, y)
# g = sns.FacetGrid(col='Segment',data=tx_user,legend_out=False)
# g.map(sns.distplot,'OverallScore')

g = sns.FacetGrid(col='Segment',data=tx_user,legend_out=False)
g.map(sns.scatterplot,'Recency','Frequency')
plt.show()

g = sns.FacetGrid(col='Segment',data=tx_user,legend_out=False)
g.map(sns.scatterplot,'Recency','Monetary')
plt.show()

tx_user.to_excel('d:\\results.xlsx')
print(tx_user.head(10))