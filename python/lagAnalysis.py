# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:46:17 2016

@author: kok
"""

from FeatureUtils import *
import matplotlib.pyplot as plt
import networkx as nx

        
        
        
y = retrieve_crime_count(2010).reshape( (77,) )
C = generate_corina_features()
popul = C[1][:,0]
cr = np.round(y / popul * 10000)

Wsp = generate_transition_SocialLag(2010)
Wsg = generate_geographical_SpatialLag_ca()

kWsp = np.zeros(Wsp.shape)


# get top k neighbors
k = 1
for i in range(len(Wsp)):
    temp = np.argpartition( -Wsp[i], k )
    kWsp[i,temp[:k]] = Wsp[i,temp[:k]]



spg = nx.Graph(kWsp)
# name nodes in graph
for nd in spg.node:
    spg.node[nd]['label'] = '{0} id{1}'.format(cr[nd], nd)
    
ccs = nx.connected_component_subgraphs(spg)

for i, subG in enumerate(ccs):    
    plt.figure()
    pos = nx.spring_layout(subG)
    nlbs = nx.get_node_attributes(subG, 'label')
    nx.draw(subG, pos, edge_color=range(len(subG.edges())), labels=nlbs,
            width=4, edge_cmap=plt.cm.Blues, with_labels=True)
    lbs = nx.get_edge_attributes(subG, 'weight')
    labels = {}
    for edge in lbs:
        labels[edge] = int(lbs[edge]*1000)
    nx.draw_networkx_edge_labels(subG, pos, edge_labels=labels)
    plt.savefig("{0}.png".format(i))


for i, row in enumerate(Wsp):
    sk = sorted(range(len(row)), key=lambda k : row[k], reverse=True)
    print i, y[i], cr[i]
#    print sk
#    print Wsp[i,sk]
#    print y[sk]
#    print cr[sk]

