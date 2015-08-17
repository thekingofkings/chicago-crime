"""
Extract the Community Area (CA) flow

1. Get reference from tract to CA from census shapefile
2. Find Chicago tracts and aggregate to CA

Author: Hongjian Wang
Date: 8/14/2015
"""


import shapefile



def generate_Tract_CA_reference():
    """
    For all tracts in Chicago, find their corresponding Community Area
    """
    sf = shapefile.Reader('../data/chicago-shp-2010/CensusTractsTIGER2010')
    
    fields = [ sf.fields[i][0] for i in range(2, len(sf.fields)) ]
    for f in fields:
        print f
        
    records = sf.records()
    
    with open('../data/chicago-tract-ca', 'w') as fout:
        for rec in records:
            fout.write('{0},{1}'.format(rec[2], rec[6]))
            fout.write('\n')
        


def get_Tract_CA_ref():
    """
    Retrieve tract to CA reference from file
    """
    tract_ca_ref = {}
    with open('../data/chicago-tract-ca', 'r') as fin:
        for line in fin:
            ls = line.split(",")    # tract ID, CA ID
            tract_ca_ref[int(ls[0])] = int(ls[1])
    return tract_ca_ref
            

if __name__ == '__main__':
    
    TC_ref = get_Tract_CA_ref()