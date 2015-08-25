"""
Parse chicago crime file
"""

import shapefile
from shapely.geometry import Polygon, Point, box



class CrimeRecord:
    
    cntBadRecord = 0    # record without GPS is bad
    cntTotal = 0
    CrimeType = ['total']

    def __init__( self, line ):
        ls = line.split(",")
        if len(ls) > 21:
            try:
                self.point = Point( float(ls[-3]), float(ls[-4]) )  # longitude, latitude
                self.id = ls[0]
                self.caseNumber = ls[1]
                self.date = ls[2]
                self.type = ls[5]
                if self.type not in CrimeRecord.CrimeType:
                    CrimeRecord.CrimeType.append(self.type)
                CrimeRecord.cntTotal += 1
            except ValueError:
                print line
                self.id = None
                CrimeRecord.cntBadRecord += 1
        else:
            self.id = None
        
        
    def __str__( self ):
        return ' '.join( [self.id, self.caseNumber, self.date, self.type, self.lat, self.lon ] )
            
        

        
class CrimeDataset:
    
    def __init__( self, fname ):
        self.f = open( fname, 'r' )
        
        
    def crimeCount_PerTract( self, tracts ):
        """
        count the number of crimes for each tract
        
        Input:
            tracts - a map of tract ID to tract object
        """
        for line in self.f:
            cr = CrimeRecord( line )
            if cr.id != None:
                for tract in tracts.values():
                    if tract.containCrime (cr):
                        tract.count['total'] += 1
                        if cr.type not in tract.count:
                            tract.count[cr.type] = 1
                        else:
                            tract.count[cr.type] += 1
                        
            
        
        
    @classmethod
    def splitFileIntoYear( cls, rawfileName ):
        """
        Split the raw file according to the year filed
        """
        cls.f = open(rawfileName, 'r')
        header = cls.f.readline()  # get rid of header line
        years = {}
        for line in cls.f:
            cr = CrimeRecord(line)
            year = cr.date[6:10]
            if year not in years:
                years[year] = open('../data/chicago-crime-{0}.csv'.format(year), 'w')
            years[year].write(line)
            
        for F in years.values():
            F.close()
            
            
   
        
        
class Tract:
    
    
    def __init__( self, shp ):
        """
        Build one Tract object from the shapefile._Shape object
        """
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.count = {'total': 0} # type: value
        
        
    
    def containCrime( self, cr ):
        """
        return true if the cr record happened within current tract
        """
        if self.bbox.contains(cr.point):
            if self.polygon.contains(cr.point):
                return True
        return False
        
        
    @classmethod
    def createAllTractObjects( cls ):
        cls.sf = shapefile.Reader('../data/chicago-shp-2010-gps/chicago_tract_wgs84')
        cls.tracts = {}
        
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            tid = cls.sf.record(idx)[3]
            trt = Tract(shp)
            cls.tracts[tid] = trt
        
        return cls.tracts
            
            
            
        
if __name__ == '__main__':
    
#    CrimeDataset.splitFileIntoYear('../data/Crimes_-_2001_to_present.csv')
    year = 2001
    
    c = CrimeDataset('../data/chicago-crime-{0}.csv'.format(year))
    T = Tract.createAllTractObjects()
    c.crimeCount_PerTract(T)    
    CrimeRecord.CrimeType.sort()
    cntKey = CrimeRecord.CrimeType
    print len(cntKey), cntKey
        
    with open('../data/chicago-crime-tract-level-{0}.csv'.format(year), 'w') as fout:
        for k, v in T.items():
            cntstr = []
            for tp in cntKey:
                if tp in v.count:
                    cntstr.append( str(v.count[tp]) )
                else:
                    cntstr.append('0')
            fout.write(','.join( [k] + cntstr ))
            fout.write("\n")
            
    print "Bad records: {0}".format(CrimeRecord.cntBadRecord)
    print "Total records: {0}".format(CrimeRecord.cntTotal)          
