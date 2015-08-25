# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:00:24 2015

@author: Hongjian

Build tract level flow from LEHD dataset.
Dependency:
    Use the LEHD-download.py script first to extract all the block level OD data.
    
This task is memory consuming. On my 8GB laptop it cannot be executed.
Rewrite in Groovy.
"""

import sys
import os


def mergeBlockCensus( a , b ):
    if len(a) != len(b):
        print "Two list length do not match!"
        return None
        
    for idx, val in enumerate(a):
        a[idx] += b[idx]
        
    return a
    
    
    
def savePairWiseTractsFeatures( fname, tracts ):
    """
    The tracts is a dictionary of dictionary, where pairwise tracts flow features are stored
    """
    # write out tract-tract flows
    with open(fname, 'w') as fout:
        for org, dict_dst in tracts.items():
            for dst, counts in dict_dst.items():
                counts_str = [ str(x) for x in counts ]
                l = ','.join([str(org), str(dst)] + counts_str)
                fout.write(l)
                fout.write('\n')
                
    

if __name__ == '__main__':
    
    print 'Warning: you need more than 8GB memroy to run this script. 16GB is recommended.'
    
    arguments = {}
    if len(sys.argv) % 2 == 1:
        for i in range(1, len(sys.argv), 2):
            arguments[sys.argv[i]] = sys.argv[i+1]
    else:
        print """Usage: TractFlow.py [options] [value]
        Possible options:
            year       e.g. 2010 default '2010'"""
            
            
    year = 2010
    if 'year' in arguments:
        year = arguments['year']
        
    
    tracts = {}
    dirPath = '../data/{0}'.format(year)
    files = os.listdir(dirPath)
    cnt = 0
    foutName = '../data/state_all_tract_level_od_JT00_{0}.csv'.format(year)
    
    if os.path.exists(foutName):
        print 'The year {0} is already merged.\nQuit Program'.format(year)
        sys.exit(0)
    
    for fn in files:
        f = open(os.path.join(dirPath, fn), 'r')
        cnt += 1
        # get rid of header line
        next(f)
        for line in f:
            ls = line.split(",")    # ls length 13
            tract_org = long(ls[0]) / 10000
            tract_dst = long(ls[1]) / 10000
            # get count S000 SA01 SA02 SA03 SE01 SE02 SE03 SI01 SI02 SI03
            counts = []
            for s in ls[2:-1]:
                counts.append( int(s) )
            
            if tract_org in tracts:
                if tract_dst in tracts[tract_org]:
                    mergeBlockCensus(tracts[tract_org][tract_dst], counts)
                else:
                    tracts[tract_org][tract_dst] = counts
            else:
                tracts[tract_org] = {}
                tracts[tract_org][tract_dst] = counts
                
        f.close()
        if cnt % 5 == 0:
            print '{0} out of {1} files processed.'.format(cnt, len(files))
            
            
    savePairWiseTractsFeatures(foutName, tracts)
    
            
    
                        


