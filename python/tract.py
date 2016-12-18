#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 16:33:22 2016

@author: hj
"""

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, box
import shapefile
import os
here = os.path.dirname(os.path.abspath(__file__))

        
class Tract:
    def __init__( self, shp, rec=None ):
        """
        Build one Tract object from the shapefile._Shape object
        """
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.count = {'total': 0} # type: value
        self.timeHist = {'total': np.zeros(24)}
        if rec != None:
            self.CA = rec[7]
        
        
    
    def containCrime( self, cr ):
        """
        return true if the cr record happened within current tract
        """
        if self.bbox.contains(cr.point):
            if self.polygon.contains(cr.point):
                return True
        return False
        
        
    
    def plotTimeHist(self, keys=None):
        """
        Plot the crime time histogram
        """
        if len(self.timeHist) == 1:
            return
        else:
            if keys is None:
                keys = self.timeHist.keys()
            values = [self.timeHist[key] for key in keys]
            
            plt.figure()
            for val in values:
                plt.plot(val)
            plt.legend(keys)
            plt.show()
            
            
        
        
    @classmethod
    def createAllTractObjects( cls ):
        cls.sf = shapefile.Reader(here + '/../data/Census-Tracts-2010/chicago-tract')
        cls.tracts = {}
        
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            rec = cls.sf.record(idx)
            tid = int(rec[2])
            trt = Tract(shp, rec)
            cls.tracts[tid] = trt
        
        return cls.tracts
            
            
            
            
    @classmethod
    def createAllCAObjects( cls ):
        cls.casf = shapefile.Reader(here + '/../data/ChiCA_gps/ChiCaGPS')
        cls.cas = {}
        
        shps = cls.casf.shapes()
        for idx, shp in enumerate(shps):
            tid = cls.casf.record(idx)[4]
            trt = Tract(shp)
            cls.cas[int(tid)] = trt
            
        return cls.cas


    @classmethod
    def visualizeRegions(cls, residence=[], nightlife=[], professional=[]):
        if hasattr(cls, "cas"):
            r = cls.cas
        elif hasattr(cls, "tracts"):
            r = cls.tracts
            
        from descartes import PolygonPatch
        f = plt.figure()
        ax = f.gca()
        for k, s in r.items():
            if k in residence:
                clr = "blue"
                p = s.polygon.centroid
                ax.text(p.x, p.y, str(k))
            elif k in nightlife:
                clr = "red"
                p = s.polygon.centroid
                ax.text(p.x, p.y, str(k))
            elif k in professional:
                clr = "green"
                p = s.polygon.centroid
                ax.text(p.x, p.y, str(k))
            else:
                clr = "white"
            ax.add_patch(PolygonPatch(s.polygon, alpha=0.5, fc=clr))
        ax.axis("scaled")
        plt.show()
        

        
        
if __name__ == "__main__":
    import sys
    Tract.createAllTractObjects()
    
    if len(sys.argv) > 1 and sys.argv[1] == "tractProfile":
        rsd = [330100, 32100, 63302, 80100, 60800]
        nl = [81700, 62200, 832000, 243400]
        pf = [839100, 320100, 81800, 760801, 81401]
        Tract.visualizeRegions(residence=rsd, nightlife=nl, professional=pf)
    else:
        Tract.visualizeRegions()
    