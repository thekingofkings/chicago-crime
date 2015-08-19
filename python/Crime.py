"""
Parse chicago crime file
"""


class CrimeRecord:

    def __init__( self, line ):
        ls = line.split(",")
        self.id = ls[0]
        self.caseNumber = ls[1]
        self.date = ls[2]
        self.type = ls[5]
        self.lat = ls[-4]
        self.lon = ls[-3]
        
        
    def __str__( self ):
        return ' '.join( [self.id, self.caseNumber, self.date, self.type, self.lat, self.lon ] )

        
class CrimeDataset:
    
    def __init__( self, fname ):
        self.f = open( fname, 'r' )
        header = self.f.readline()
        
        
    def splitFileIntoYear( self ):
        """
        Split the file according to the year filed
        """        
        years = {}
        for line in self.f:
            cr = CrimeRecord(line)
            year = cr.date[6:10]
            if year not in years:
                years[year] = open('../data/chicago-crime-{0}.csv'.format(year), 'w')
            years[year].write(line)
            
        for F in years.values():
            F.close()
            
            
  
        
        
if __name__ == '__main__':
    
    c = CrimeDataset('../data/Crimes_-_2001_to_present.csv')
        
    c.splitFileIntoYear()