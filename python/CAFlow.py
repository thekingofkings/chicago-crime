"""
Extract the Community Area (CA) flow

1. Get reference from tract to CA from census shapefile
2. Find Chicago tracts and aggregate to CA

Author: Hongjian Wang
Date: 8/14/2015
"""

import sys
import shapefile
from TractFlow import mergeBlockCensus, savePairWiseTractsFeatures


"""
reuse the mergeBlockCensus function, which takes the sum of two list
"""
mergeCACensus = mergeBlockCensus


"""
reuse the savePairWiseTractsFeatures function, which save dictionary of dictionary into file
"""
savePairWiseCAFeatures = savePairWiseTractsFeatures



def saveNodalCAFeatures(foutName, CA_features):
    """
    The CA_features is a dicationary
    """
    with open(foutName, 'w') as fout:
        for caID, features in CA_features.items():
            fstr = [str(x) for x in features]
            fout.write( ",".join( [str(caID)] + fstr) )
            fout.write("\n")
            
            

def generate_Tract_CA_reference():
    """
    For all tracts in Chicago, find their corresponding Community Area
    """
    sf = shapefile.Reader('../data/chicago-shp-2010-gps/chicago_tract_wgs84')
    
    fields = [ sf.fields[i][0] for i in range(1, len(sf.fields)) ]
    for f in fields:
        print f
        
    records = sf.records()
    
    with open('../data/chicago-tract-ca', 'w') as fout:
        for rec in records:
            fout.write('{0},{1}'.format(rec[3], rec[6]))
            fout.write('\n')
            
    return sf
        


def get_Tract_CA_ref():
    """
    Retrieve tract to CA reference from file
    
    Return value:
        A Map from tract ID to CA ID, both are integer
    """
    tract_ca_ref = {}
    with open('../data/chicago-tract-ca', 'r') as fin:
        for line in fin:
            ls = line.split(",")    # tract ID, CA ID
            tract_ca_ref[int(ls[0])] = int(ls[1])
    return tract_ca_ref



def tract_CA_flowAggregate(finName, foutName):
    """
    aggregate flow information from tract to CA
    flow format:
        srcID, dstID, f1, f2, ...
    """
    TC_ref = get_Tract_CA_ref()
    
    CA_census = {}
    
    with open(finName) as fin:
        for line in fin:
            ls = line.split(",")
            src_tract = int(ls[0])
            dst_tract = int(ls[1])
            
            
            if src_tract in TC_ref and dst_tract in TC_ref:
                d = []
                src_ca = TC_ref[src_tract]
                dst_ca = TC_ref[dst_tract]
                for val in ls[2:]:
                    d.append(int(val))
                    
                if src_ca in CA_census:
                    if dst_ca in CA_census[src_ca]:
                        CA_census[src_ca][dst_ca] = mergeCACensus(CA_census[src_ca][dst_ca], d)
                    else:
                        CA_census[src_ca][dst_ca] = d
                else:
                    CA_census[src_ca] = {}
                    CA_census[src_ca][dst_ca] = d
                        
    
    savePairWiseCAFeatures(foutName, CA_census)    
    return CA_census



def tract_CA_nodalFeatureAggregate(finName, foutName):
    """
    Aggregate nodal feature from tract to CA
    feature format:
        srcID, f1, f2
    """
    TC_ref = get_Tract_CA_ref()
    
    CA_features = {}
    
    with open(finName) as fin:
        for line in fin:
            ls = line.split(",")
            tracID = int(ls[0])
            
            if tracID in TC_ref:
                d = []
                caID = TC_ref[tracID]
                for val in ls[1:]:
                    d.append(int(val))
                    
                if caID in CA_features:
                    mergeCACensus(CA_features[caID], d)
                else:
                    CA_features[caID] = d
                    
                    
    saveNodalCAFeatures(foutName, CA_features)
    return CA_features


if __name__ == '__main__':
    
#    sf = generate_Tract_CA_reference()
    args = {}
    
    if len(sys.argv) % 2 == 1:
        for i in range(1, len(sys.argv), 2):
            args[sys.argv[i]] = sys.argv[i+1]   # year 2010 # datatype (lehd/crime)

    
    finName = '../data/state_all_tract_level_od_JT00_{0}.csv'.format(args['year'])
    foutName = '../data/chicago_ca_od_{0}.csv'.format(args['year'])
    if 'datatype' in args:
        if args['datatype'] == 'crime':
            finName = '../data/chicago-crime-tract-level-{0}.csv'.format(args['year'])
            foutName = '../data/chicago-crime-ca-level-{0}.csv'.format(args['year'])
            r = tract_CA_nodalFeatureAggregate(finName, foutName)
        elif args['datatype'] == 'lehd':    
            finName = '../data/state_all_tract_level_od_JT00_{0}.csv'.format(args['year'])
            foutName = '../data/chicago_ca_od_{0}.csv'.format(args['year'])
            tract_CA_flowAggregate(finName, foutName)
    
