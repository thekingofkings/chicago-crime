from flask import Flask, request, jsonify
from flask import render_template
from NBRegression import *

import os
here = os.path.dirname(os.path.abspath(__file__))


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
    print 'crime type', crimeT, '\n'

    permutationTest_onChicagoCrimeData(year=year, features=features, 
            logFeatures=logF, crimeType=crimeT, flowType=flowT, iters=iters)
    # print redirection ends
    sys.stdout.close()
    s = None
    with open(os.path.join(here, 'templates', '{0}'.format(fname))) as fin:
        s = fin.read().replace('\n', '<br>')
    return s


    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
