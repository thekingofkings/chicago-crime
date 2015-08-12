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


import os


def mergeBlockCensus( a , b ):
    if len(a) != len(b):
        print "Two list length do not match!"
        exit()
        
    c = []
    for idx, val in enumerate(a):
        c.append(val + b[idx])
        
    return c

if __name__ == '__main__':
    
    tracts = {}
    dirPath = '../data/2010'
    files = os.listdir(dirPath)
    cnt = 0
    
    for fn in files:
        f = open(os.path.join(dirPath, fn), 'r')
        cnt += 1
        # get rid of header line
        next(f)
        for line in f:
            ls = line.split(",")    # ls length 13
            tract_org = ls[0][:-3]
            tract_dst = ls[1][:-3]
            # get count S000 SA01 SA02 SA03 SE01 SE02 SE03 SI01 SI02 SI03
            counts = []
            for s in ls[2:-1]:
                counts.append( int(s) )
            
            if tract_org in tracts:
                if tract_dst in tracts[tract_org]:
                    tracts[tract_org][tract_dst] = mergeBlockCensus(tracts[tract_org][tract_dst], counts)
                else:
                    tracts[tract_org][tract_dst] = counts
            else:
                tracts[tract_org] = {}
                tracts[tract_org][tract_dst] = counts
                
        f.close()
        if cnt % 5 == 0:
            print '{0} out of {1} files processed.'.format(cnt, len(files))
            
            
    # write out tract-tract flows
    with open('../data/state_all_tract_level_od_JT00_2010.csv', 'w') as fout:
        for org, dict_dst in tracts.items():
            for dst, counts in dict_dst.items():
                counts_str = [ str(x) for x in counts ]
                l = ','.join([org, dst] + counts_str)
                fout.write(l)
                fout.write('\n')
        
            
    
                        


