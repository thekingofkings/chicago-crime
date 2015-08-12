

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
        header = f.readline()
  
        
        
if __name__ == '__main__':
    
    c = CrimeDataset('../data/Crimes_-_2001_to_present.csv')
        
        