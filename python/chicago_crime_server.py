from flask import Flask, request, jsonify
from flask import render_template
from NBRegression import *



app = Flask(__name__)


features = ['density', 'disadvantage', 'ethnic', 'pctblack', 'pctship',
    'population', 'poverty', 'residential', 'sociallag', 'spatiallag',
    'temporallag']


@app.route('/')
def input_parameter():
    return render_template('nb-parameter-setting.html')
    

@app.route('/set-parameter', methods=['GET'])    
def set_parameter():
    a =  request.args
   
    logF = []
    for k in a.keys():
        if a.get(k) == 'log':
            logF.append(k)
    import sys
    import time
    fname = 'file-{0}'.format(time.strftime('%m-%d-%y-%H-%M-%S'))
    sys.stdout = open('templates/' + fname, 'w')
    # every print is redirected to the file
    print 'Selected features', a.keys()
    print 'Features take log', logF, '\n'
    permutationTest_onChicagoCrimeData(year=2010, features=a.keys(), logFeatures=logF, iters=3)
    # print redirection ends
    sys.stdout.close()
    s = None
    with open('templates/{0}'.format(fname)) as fin:
        s = fin.read().replace('\n', '<br>')
    return s


    
if __name__ == '__main__':
    app.run(debug=True)