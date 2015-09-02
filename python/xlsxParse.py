# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 14:24:07 2015

@author: hxw186


extract CA level features from xlsx file.
"""

from openpyxl import *

def extractCAlevelFeature(fin = "../data/Chicago_race_place of origin _census_tract_CA_2010.xlsx", fout = '../data/Chicago-ca-race.xlsx'):
    wb = load_workbook(fin)
   
    wo = Workbook(write_only=True)
    wos = wo.create_sheet()
    
    ws = wb.active
    
    for row in ws.iter_rows():
        if row[0].row in [1, 2, 3]:
            wos.append( row )
        elif row[1].value == '070':
            wos.append( row )
            
            
    wo.save(fout)



if __name__ == '__main__':
    
#    extractCAlevelFeature("../data/Education by Race by Census Tract and Community Area.xlsx", "../data/chicago-ca-education.xlsx")
    extractCAlevelFeature("../data/Household Income by Race and Census Tract and Community Area.xlsx", "../data/chicago-ca-income.xlsx")
