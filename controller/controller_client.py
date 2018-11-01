# -*- coding: utf-8 -*-
"""

Created on November 11, 2017

@author:  neerbek

command line wrapper for controller_client_app

"""

import os
import sys
import controller_client_app

numWorkers = 1
port = 5000

def syntax():
    print("""syntax: controller_client.py
    -numWorkers <int>
    -cmd <python-file>
    -cmdDir <path>
    -port <int>
    [-h | --help | -?]

-numWorkers is the max number of client processes (workers) this controller will launch at one time
-cmd is the pythonfile that the workers should execute
-port is the port to run client on
-cmdDir is the directory from where to execute the client
""")
    sys.exit()


arglist = sys.argv
# arglist = "controller_client -numWorkers 2 -cmdDir ../../taboo-core".split(" ")
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == '-numWorkers':
        numWorkers = int(arg)
    elif setting == "-cmdDir":
        os.chdir(arg)
    elif setting == "-cmd":
        controller_client_app.runJobCmd = arg
    elif setting == "-port":
        port = int(arg)
    else:
        # expected option with no argument
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i

controller_client_app.initializeModule(numWorkers=numWorkers)

controller_client_app.app.run(port=port)
