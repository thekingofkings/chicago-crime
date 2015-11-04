import os, sys

os.environ['MPLCONFIGDIR'] = '/tmp/'
os.environ['DISPLAY'] = ':5'
sys.path.append('/data/urban-flow-analysis/python/')


from chicago_crime_server import app as application

