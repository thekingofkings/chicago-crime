import urllib2
import re
from bs4 import BeautifulSoup as bs
import subprocess
import os.path
import sys
import os



def download_OD_state_year( state_od_url, stateN, year ):
    """
    Download the origin-destination flow data for given state in given year.
    """
    
    # open state OD page
    page = urllib2.urlopen(od_url)
    content = page.read()
    od = bs(content, 'html.parser')
    
    # setup download folder by year
    downloadFolder = '../data/' + year
    if os.path.exists(downloadFolder):
        pass
    else:
        os.mkdir(downloadFolder)
        
        
    # find all files
    types = ['main', 'aux']
    for t in types:
        search_href = stateN + '_od_{0}_JT00_{1}'.format(t, year)
        f = od.find_all(href=re.compile(search_href))
        
        if len(f) > 0:
            f = f[0]
        else:
            print '{0} does not exist!'.format(search_href)
            continue
            
        downloadFilePath = os.path.join(downloadFolder, f.attrs['href'])    # .csv.gz file
        
        # download the file if does not exist
        if os.path.exists(downloadFilePath[:-3]):
            print 'File {0} exits. No need for download.'.format(downloadFilePath[:-3])
        else:
            file_url = od_url + f.attrs['href']
            fpage = urllib2.urlopen(file_url)    
            fout = open(downloadFilePath, 'wb')
            fout.write(fpage.read())
            fout.close()
            
            # decompress the *.gz file  
            subprocess.call(['7z', 'x', '-o{0}'.format(downloadFolder), os.path.abspath(downloadFilePath) ])
            
            # remove the downloaded gz file
            os.remove(downloadFilePath)
            print 'File {0} downloaded'.format(downloadFilePath)
        

if __name__ == '__main__':
    
    arguments = {}
    if len(sys.argv) % 2 == 1:
        for i in range(1, len(sys.argv), 2):
            arguments[sys.argv[i]] = sys.argv[i+1]
    else:
        print """Usage: LEHD-download.py [options] [value]
        Possible options:
            stateName  e.g. pa, default value 'all'
            year       e.g. 2010 default '2010'"""
                    
    print arguments
        
    
    url = 'http://lehd.ces.census.gov/data/lodes/LODES7/'
    page = urllib2.urlopen(url)
    content = page.read()
    
    domtree = bs(content, 'html.parser')
    
    # go into each state
    imgs = domtree.find_all('img', attrs={'src': '/icons/folder.gif'})
    
    states = []
    
    for img in imgs:
        state = img.parent.next_sibling.contents[0]
        states.append(state.string[0:2])
        
    stateN = arguments['stateName'] if 'stateName' in arguments else 'all'
    year = arguments['year'] if 'year' in arguments else '2010'
        
    
    if stateN in states:
        # Origin - Destination url
        od_url = url + stateN + '/od/'
        download_OD_state_year( od_url, stateN, year )
    elif stateN == 'all':
        for s in states:
            od_url = url + s + '/od/'
            download_OD_state_year( od_url, s, year )
    else:
        print 'State is not found!'
        
    
    