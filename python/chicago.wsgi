import os, sys

os.environ['MPLCONFIGDIR'] = '/tmp/'
os.environ['DISPLAY'] = ':5'
os.environ['LD_LIBRARY_PATH'] = '/opt/glibc-2.14/lib'
sys.path.append('/data/urban-flow-analysis/python/')


from chicago_crime_server import app as application

