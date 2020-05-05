import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from networkHelpder import draw_graph

try:
    df = pd.read_excel('Data\Data_Sales_SKU_CUST_HIERARCHY.xlsx')
except Exception as ex:
    print(ex)
    exit(1)
df['ID'] = df['ID'].astype('str')
market_basket = df.groupby(['ID','HIERARCHY'])['QUAN']
market_basket = market_basket.sum().unstack().reset_index().fillna(0).set_index('ID')
def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    if datapoint >= 1:
        return 1
market_basket = market_basket.applymap(encode_data)
market_basket.fillna(0, inplace = True)
itemsets = apriori(market_basket, min_support=0.02, use_colnames=True)
print(itemsets)
rules = association_rules(itemsets, metric="lift", min_threshold=0.5)


    
draw_graph(rules, 6)
print(rules)