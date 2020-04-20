import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df = pd.read_excel('Data_Sales_SKU_CUST.xlsx')
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
itemsets = apriori(market_basket, min_support=0.01, use_colnames=True)
rules = association_rules(itemsets, metric="lift", min_threshold=0.5)
print(rules)