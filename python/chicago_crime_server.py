from flask import Flask, request, jsonify, redirect
from flask import render_template, send_from_directory
from NBRegression import *

import os
here = os.path.dirname(os.path.abspath(__file__))

import glob


app = Flask(__name__)
app.debug = True


features = ['density', 'disadvantage', 'ethnic', 'pctblack', 'pctship',
    'population', 'poverty', 'residential', 'sociallag', 'spatiallag',
    'temporallag']


@app.route('/')
def input_parameter():
    return render_template('nb-parameter-setting.html')
    
    

@app.route('/set-parameter', methods=['GET'])    
def set_parameter():
    a =  request.args

    
    features = list(a.keys())
    crimeT = a.get('crimeT')
    if crimeT == 'total':
        crimeT = ['total']
    elif crimeT == 'violent':
        crimeT = ['HOMICIDE', 'CRIM SEXUAL ASSAULT', 'BATTERY', 'ROBBERY', 
                'ARSON', 'DOMESTIC VIOLENCE', 'ASSAULT']

    flowT = int(a.get('flowT'))
    iters = int(a.get('iters'))
    year = int(a.get('year'))
  
    features.remove('crimeT')
    features.remove('flowT')
    features.remove('iters')
    features.remove('year')

    logF = []
    for k in a.keys():
        if a.get(k) == 'log':
            logF.append(k)
        elif a.get(k) == 'none':
            features.remove(k)

    import sys
    import time
    import os
    fname = 'file-{0}'.format(time.strftime('%m-%d-%y-%H-%M-%S'))
    sys.stdout = open(os.path.join(here, 'templates', fname), 'w')
    # every print is redirected to the file
    print 'Selected features', features 
    print 'Features take log', logF
    print 'Year', year
    print 'Flow type:', flowT, '(0 - total, 4 - low income)'
    print 'crime type', crimeT
    print 'number of iterations', iters, '\n'

    permutationTest_onChicagoCrimeData(year=year, features=features, 
            logFeatures=logF, crimeType=crimeT, flowType=flowT, iters=iters)
    # print redirection ends
    sys.stdout.close()
    s = None
    return redirect('result/' + fname)



@app.route('/history')
def list_previous_results():
    s = glob.glob(here + '/templates/file*')    
    items = []
    for f in s:
        with open(f, 'r') as fin:
            head = [next(fin) for x in range(6)]
        fn = os.path.basename(f)
        item = {'head': head, 'name': fn}
        items.append(item)
    return render_template('history.html', items=items) 



@app.route('/result/<fname>')
def format_result(fname):
    fn = here + '/templates/' + fname
    with open(fn, 'r') as fin:
        head = [fin.readline() for x in range(6)]
        fin.readline() 
        key = fin.readline().strip()
        rows = [] 
        while (key != ''):
            values = fin.readline().split(" ")
            row = {'key': key, 'values': values}
            rows.append(row)
            key = fin.readline().strip()
    return render_template('result.html', head=head, rows = rows)

        
        
        
        
@app.route('/nb-permute')
def nb_permute():
    return render_template('nb-permute.html')
    
    
    
@app.route('/new-permute')
def new_permute():
    
    a =  request.args
    year = int(a.get('year')) if a.get('year') != '' else '2010'
    iters = a.get('iters') if a.get('iters') != '' else 10
    lags = []
    lags.append( "1" if "social-lag-crime" in a else "0" )
    lags.append( "1" if "spatial-lag-crime" in a else "0" )
    lags.append( "1" if "social-lag-disadv" in a else "0" )
    lags.append( "1" if "spatial-lag-disadv" in a else "0" )
    lagsFlag = "".join( lags )
    
    ep = 'exposure' if 'exposure' in a else 'noexposure'
    
    print a
    print lagsFlag, iters, ep, year
    
    fname = "glmmadmb--totallehd-totalcrime-bysource-{0}-logpop-{1}-{2}-logpopdensty-.out".format(ep, lagsFlag, iters)
    coefficients_pvalue(lagsFlag, itersN=iters, exposure=ep, year=year)    
    
    return redirect('download/' + fname)


@app.route('/download/<fname>')
def download_result(fname):
    fn = here + '/../R/'
    return send_from_directory(fn, fname, cache_timeout=0)
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
