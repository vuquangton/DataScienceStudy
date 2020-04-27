import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  
import random
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
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

def draw_graph(rules, rules_to_show):
    G1 = nx.DiGraph()
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
    strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11']   
    for i in range (rules_to_show):    
        G1.add_nodes_from(["R"+str(i)])    
        for a in rules.iloc[i]['antecedents']:
            G1.add_nodes_from([a])
            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 2)
       
        for c in rules.iloc[i]['consequents']:
            G1.add_nodes_from([c])
            G1.add_edge("R"+str(i), c, color=colors[i],  weight=2)
 
    for node in G1:
       found_a_string = False
       for item in strs: 
           if node==item:
                found_a_string = True
       if found_a_string:
            color_map.append('yellow')
       else:
            color_map.append('green')       
 
   
    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]
 
    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
   
    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    
    nx.draw_networkx_labels(G1, pos)
    plt.show()
    
draw_graph(rules, 6)
print(rules)