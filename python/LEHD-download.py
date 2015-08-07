import urllib2
import re
from bs4 import BeautifulSoup as bs
import subprocess
import os.path


url = 'http://lehd.ces.census.gov/data/lodes/LODES7/'
page = urllib2.urlopen(url)
content = page.read()

domtree = bs(content, 'html.parser')

# go into each state
imgs = domtree.find_all('img', attrs={'src': '/icons/folder.gif'})

for img in imgs:
    state = img.parent.next_sibling.contents[0]
    stateN = state.string[0:2]
    
    if stateN == 'il':
        # Origin - Destination folder
        od_url = url + state.attrs['href'] + 'od/'
        print od_url
        page = urllib2.urlopen(od_url)
        content = page.read()
        od = bs(content, 'html.parser')
        
        # find all files
        f = od.find_all(href=re.compile(stateN + '_od_main_JT00_2010'))[0]
        
        file_url = od_url + f.attrs['href']
        fpage = urllib2.urlopen(file_url)
        fout = open('../data/' + f.attrs['href'], 'wb')
        fout.write(fpage.read())
        fout.close()
        
        # decompress the *.gz file
        subprocess.call(['7z', 'x', os.path.abspath('../data/' + f.attrs['href']) ])
        
    
    