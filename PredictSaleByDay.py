import pandas as pd
import numpy as np
import datetime as dt
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df_source = pd.read_excel('Data\Data_Sales_SKU_CUST.xlsx')
df_saleByDay = df_source.groupby('INVOICEDATE')['VALUE'].sum().reset_index()
df_saleByDay['INVOICEDATE'] = pd.to_datetime(df_saleByDay['INVOICEDATE'])
df_saleByDay['INVOICEDATE']=df_saleByDay['INVOICEDATE'].map(dt.datetime.toordinal)



y = np.asarray(df_saleByDay['VALUE'])
X = df_saleByDay[['INVOICEDATE']]
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3,random_state=0)         
print(X_test)

model = LinearRegression() #create linear regression object
model.fit(X_train, y_train) #train model on train data
#model.score(X_train, y_train) #check score
try:
 
    y_pred = model.predict(X_test)
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    print(df)
    pyplot.scatter(X_test, y_test,  color='gray')
    pyplot.plot(X_test, y_pred, color='red', linewidth=2)
    pyplot.show()
except Exception as identifier:
    print(identifier)





