import pandas as pd
from sqlConnect import MSSQLHelper
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


try:
    #df = pd.read_excel('Data\Data_Sales_SKU_CUST.xlsx')
    df = MSSQLHelper.read_data()
    print(df.head(10))
except Exception as ex:
    print(ex)
    exit(1)

#df['ID'] = df['ID'].astype('str')
df['TX_NO'] = df['TX_NO'].astype('str')

market_basket = df.groupby(['TX_NO','SKU'])['QUAN']
market_basket = market_basket.sum().unstack().reset_index().fillna(0).set_index('TX_NO')
def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    if datapoint >= 1:
        return 1

market_basket = market_basket.applymap(encode_data)
market_basket.fillna(0, inplace = True)
itemsets = apriori(market_basket, min_support=0.03, use_colnames=True)
print(itemsets)
rules = association_rules(itemsets, metric="lift", min_threshold=0.7)
print(rules)