#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Parse Chicago house price per sqft.

Created on Fri May 19 20:30:20 2017

@author: hxw186
"""

from tract import Tract
import pandas as pd
from shapely.geometry import Point
import pickle



def get_individual_house_price():
    houses = pd.read_csv("..//data/house_source.csv", index_col=0)
    houses = houses.loc[lambda x : (x["priceSqft"] > 30) & (x["priceSqft"] < 3000), :]
    return houses


def retrieve_CA_avg_house_price():
    houses = get_individual_house_price()
    
    cas = Tract.createAllCAObjects()
    house_cnt = {k:0 for k in cas.keys()}
    avg_price = {k:0.0 for k in cas.keys()}
    
    for idx, house in houses.iterrows():
        p = Point(house.lon, house.lat)
        for k, ca in cas.items():
            if ca.polygon.contains(p):
                house_cnt[k] += 1
                avg_price[k] += house.priceSqft
                break
    
    for k in house_cnt.keys():
        if house_cnt[k] == 0:
            print k
        else:
            avg_price[k] /= house_cnt[k]
    
    assert avg_price[54] == 0
    avg_price[54] = (avg_price[53] + avg_price[55]) / 2
    
    with open("../data/ca-average-house-price.pickle", 'w') as fout:
        pickle.dump(avg_price, fout)
        pickle.dump(house_cnt, fout)
        
    return avg_price



if __name__ == '__main__':
    avg_price = retrieve_CA_avg_house_price()