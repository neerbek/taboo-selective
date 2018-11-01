# -*- coding: utf-8 -*-
"""

Created on November 25, 2017

@author:  neerbek

command line wrapper for controller_server_app
"""

# curl http://127.0.0.1:5001/get_current_log?jobname=exp137 | python3 -m json.tool

import sys
import time
from typing import List, Tuple
import io
import os
import json

import controller
import clients

jobfile = "jobfile.json"
donefile = None
moveToSvn = False
loopTimer = 5.0

def syntax():
    print("""syntax: controller_server.py |
    -jobfile <file> |
    -donefile <file> |
    -loopTimer <float> |
    -moveToSvn |
    [-h | --help | -?]

-jobfile contains an jsonencoded list of jobs and parameters to run
-donefile is file to save list of completed jobs. If none given then <jobfile>_done.<jobfile.ending>
-loopTimer seconds to wait between updating client info
-moveToSvn signals to store results in svn
""")
    sys.exit()


arglist = sys.argv
argn = len(arglist)

i = 1
# if argn == 1:
#     syntax()

print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == "-jobfile":
        jobfile = arg
    elif setting == "-donefile":
        donefile = arg
    elif setting == "-loopTimer":
        loopTimer = float(arg)
    else:
        # expected option with no argument
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        elif setting == '-moveToSvn':
            moveToSvn = True
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i

if jobfile == None:
    print("missing parameters")
    syntax()

if donefile == None:
    index = len(jobfile)
    if jobfile.rfind(".") != -1:
        index = jobfile.rfind(".")
    donefile = jobfile[:index] + "_done" + jobfile[index:]

def updateJobList(clientList: List[controller.Client]):
    c = controller.Client(None, None, None)
    for c in clientList:
        c.updateJobList()
        c = controller.Client(None, None, None)  # type hint

def storeCompletedJobs(clientList: List[controller.Client]):
    for c in clientList:
        print("for server: " + c.name)
        j: controller.RunningJob
        for j in c.jobs:
            print(j)
            if isinstance(j, controller.RunningJob) and j.isCompleted:
                print("removing job: " + j.jobid)
                if not c.cleanJob(j):
                    raise Exception("falied to clean job: " + j.jobid)
                if moveToSvn:
                    if not c.storeJob(j):
                        raise Exception("falied to clean job: " + j.jobid)

def assignNewJobs(clientList: List[controller.Client], jobList: List[controller.Job], doneList: List[controller.Job]) -> Tuple[List[controller.Job], List[controller.Job]]:
    remainingJobList = []
    for j in jobList:
        jobAdded = False
        for c in clientList:
            if c.emptySlots == 0:
                continue
            if c.submitJob(j):
                c.emptySlots -= 1
                print("client " + c.__repr__() + " added job" + j.__repr__())
                jobAdded = True
            else:
                print("failed to add job" + j.__repr__() + " on client " + c.__repr__())
        if not jobAdded:
            remainingJobList.append(j)
        else:
            doneList.append(j)
    return (remainingJobList, doneList)

def toJSON(e):
    return json.dumps(e, default=lambda o: vars(o))

def loadJobList(jobfile):
    res = ""
    if not os.path.isfile(jobfile):
        return []

    with io.open(jobfile, 'r', encoding='utf8') as f:
        for line in f:
            res += line  # include \n
    return controller.Jobs.demarshal(res)

def saveJobList(jobfile, joblist):
    with io.open(jobfile, 'w', encoding='utf8') as f:
        f.write("[\n")
        for i in range(len(joblist)):
            j = joblist[i]
            f.write(toJSON(j))
            if i < len(joblist) - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("]\n")


clientList: List[controller.Client]; clientList = clients.clientList
jobList = loadJobList(jobfile)
doneList = loadJobList(donefile)

# if len(jobList) != 0:
#     print("Has jobs")
#     for j in jobList:
#         print(j)

if len(clientList) == 0:
    print("No servers given. Exiting")
    sys.exit(0)

# print(clientList)
print("looping for first time")
while True:
    updateJobList(clientList)
    storeCompletedJobs(clientList)
    remainingJobs = len(jobList)
    (jobList, doneList) = assignNewJobs(clientList, jobList, doneList)
    if len(jobList) != remainingJobs:
        print("saving files")
        saveJobList(jobfile, jobList)
        saveJobList(donefile, doneList)
    if len(jobList) == 0:
        updateJobList(clientList)  # have to update the job list the first time we arrive here
        isDone = True
        for c in clientList:
            if len(c.jobs) != 1:
                isDone = False
                break  # still waiting for result
        if isDone:
            break  # all jobs done
    print("sleeping...", flush=True)
    time.sleep(loopTimer)

print("Done")

# jobList = []
# jobList.append(controller.Job("job001", ["-port", "5002"]))
# jobList.append(controller.Job("job002", ["-port", "5002"]))
#
#
# j = toJSON(jobList)
