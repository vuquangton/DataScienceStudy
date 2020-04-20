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
df = pd.read_excel('Data_KHTT_profile.xlsx',converters={'NgayCapNhatCuoi':pd.to_datetime}, index_col=0) 

tx_user = pd.DataFrame(df['MaThe7'].unique())
tx_user.columns = ['MaThe7']

tx_revenue = df.groupby('MaThe7').DoanhSo.sum().reset_index()

tx_user = pd.merge(tx_user, tx_revenue, on='MaThe7')
#plot the histogram
plot_data = [
    go.Histogram(
        x=tx_user['DoanhSo']
    )
]

plot_layout = go.Layout(
        title='DoanhSo'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.plot(fig)


#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['DoanhSo']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['DoanhSo']])

#order the frequency cluster
tx_user = order_cluster('FrequencyCluster', 'DoanhSo',tx_user,True)

#see details of each cluster 
print(tx_user.groupby('FrequencyCluster')['DoanhSo'].describe() )