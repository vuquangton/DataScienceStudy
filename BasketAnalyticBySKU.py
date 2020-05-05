import os, sys

os.environ["MATPLOTLIBDATA"] = os.path.join(os.path.split(sys.executable)[0], "Lib\\site-packages\\matplotlib\\mpl-data")
print(os.environ["MATPLOTLIBDATA"])

import pandas as pd
from sqlConnect import MSSQLHelper
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from networkHelpder import draw_graph

try:
    df = pd.read_excel('Data\Data_Sales_SKU_CUST.xlsx')
    #df = MSSQLHelper.read_data()
    print(df.head(10))
except Exception as ex:
    print(ex)
    exit(1)

#df['ID'] = df['ID'].astype('str')
df['ID'] = df['ID'].astype('str')

market_basket = df.groupby(['ID','SKU'])['QUAN']
market_basket = market_basket.sum().unstack().reset_index().fillna(0).set_index('ID')
def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    if datapoint >= 1:
        return 1

market_basket = market_basket.applymap(encode_data)
market_basket.fillna(0, inplace = True)

#the items and itemsets with at least 60% support:
itemsets = apriori(market_basket, min_support=0.01, use_colnames=True)
print(itemsets)
#rules = association_rules(itemsets, metric="lift", min_threshold=0.7)
print(itemsets)


association_rules(itemsets, metric="confidence", min_threshold=0.7)
rules = association_rules(itemsets, metric="lift", min_threshold=0.7)
print(rules)
#print(rules.apply(lambda x: ";".join(x.Pairs),axis=1).to_csv("D:\\rules.csv",index=False))
#5rules.to_csv('D:\\rules.csv', index=False)
draw_graph(rules, 4)