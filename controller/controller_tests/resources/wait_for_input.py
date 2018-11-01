# -*- coding: utf-8 -*-
"""

Created on November 25, 2017

@author:  neerbek
"""

import sys
from flask import Flask  # type: ignore
from flask import request

app = Flask(__name__)

port = 5001

arglist = sys.argv
argn = len(arglist)

i = 1
print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == '-port':
        port = int(arg)
    else:
        msg = "unknown option: " + setting
        print(msg)
        next_i = i + 1
    i = next_i

@app.route("/ready")
def ready():
    return "[\"ok\"]"

@app.route("/stop")
def stop():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return "[\"ok\"]"


if __name__ == "__main__":
    print("running on port ", port)
    print(":{}/stop to terminate program".format(port), flush=True)  # need to flush here, ow we only see this msg when app terminates

    app.run(port=port)
