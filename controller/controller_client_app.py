# -*- coding: utf-8 -*-
"""

Created on November 25, 2017

@author:  neerbek
"""

import time
from flask import Flask    # type: ignore
from flask import request
from flask import Response
import json
import os
import subprocess
from typing import List, Any

import worker
import controller

app = Flask(__name__)
workers: List[worker.Worker]; workers = []

workerManager = worker.WorkerManager()

def initializeModule(numWorkers):
    workerManager.initialize(numWorkers)

# Error class - http://flask.pocoo.org/docs/0.12/patterns/apierrors/
class InvalidUsage(Exception):
    def __init__(self, message: str, status_code: int=400, payload: dict=None) -> None:
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['type'] = type(self)
        rv['message'] = self.message
        return rv

# Error handler
@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error: InvalidUsage) -> Response:  # type:ignore
    return handle_invalid_usage_impl(error)

def handle_invalid_usage_impl(error: InvalidUsage) -> Response:
    msg = toJSON(error.to_dict)
    return response_error(msg, error.status_code)

@app.errorhandler(worker.NoWorkerException)
def handle_invalid_no_worker(error: worker.NoWorkerException) -> Response:  # type:ignore
    return handle_invalid_worker_usage_impl(error)

@app.errorhandler(worker.JobStillRunningException)
def handle_invalid_job_still_running(error: worker.JobStillRunningException) -> Response:  # type:ignore
    return handle_invalid_worker_usage_impl(error)

def handle_invalid_worker_usage_impl(error: worker.ExceptionWithMessage) -> Response:
    vals = {"type": str(type(error)), "message": error.message}
    msg = toJSON(vals)
    return response_error(msg, 400)

# for checking connection
# http://localhost:5000/hello_world
@app.route("/hello_world")
def hello_world():  # type: ignore
    return hello_world_impl()

def hello_world_impl() -> Response:
    pres = ["here", "is", "answer"]
    return response_success(pres)

@app.route("/has_empty_worker")
def has_empty_worker():  # type: ignore
    return has_empty_worker_impl()

def has_empty_worker_impl() -> Response:
    boolValue = workerManager.hasReadyWorker()
    # could return REST resource not available ...
    return response_success([boolValue])

@app.route(controller.Client.JobListCmd)
def getJobList():  # type: ignore
    return getJobListImpl()

def getJobListImpl() -> Response:
    print('getJobList start')
    jobs = workerManager.getJobList()
    return response_success(jobs)

@app.route(controller.Client.SubmitJobCmd)
def run_job_cmd():  # type: ignore
    return run_job_cmd_impl()


runJobCmd = "../taboo-jan/functionality/kmeans_cluster_cmd3.py"
def run_job_cmd_impl() -> Response:
    print('submit job start')
    jobname = request.args.get('jobname')
    params = request.args.getlist('params')
    if is_none([jobname]) or params is None or len(params) == 0:
        return response_error("submit job expects parameters jobname, params")
    print("params ", params)
    w = workerManager.findReadyWorker()
    if w is None:
        return response_error("submit job: no workers available")
    if not os.path.exists(runJobCmd):
        return response_error("command " + runJobCmd + " not found from dir " + os.getcwd())
    p = ["OMP_NUM_THREADS=2", "ipython3", runJobCmd, "--"]
    for param in params:
        p.append(param)
    p.append("-filePrefix")
    p.append("save_" + jobname)
    w.addNew(jobname, p)
    w.setOutput(run_command(p, worker.Worker.getLogfile(jobname)))
    return response_success(["ok"])

@app.route("/get_current_log")
def get_current_log():  # type: ignore
    return get_current_log_impl()

def get_current_log_impl() -> Response:
    print('get_current_log start')
    jobname = request.args.get('jobname')
    if is_none([jobname]):
        return response_error("get_last_log expects parameters jobname")
    w = workerManager.getWorker(jobname)
    log = w.getCurrentLog()
    return response_success(log)

@app.route("/get_last_log")
def get_last_log():  # type: ignore
    return get_last_log_impl()

def get_last_log_impl() -> Response:
    print('get_last_log start')
    jobname = request.args.get('jobname')
    if is_none([jobname]):
        return response_error("get_last_log expects parameters jobname")
    w = workerManager.getWorker(jobname)
    log = w.getLastLog()
    return response_success(log)

@app.route("/is_worker_done")
def is_worker_done():  # type: ignore
    return is_worker_done_impl()

def is_worker_done_impl() -> Response:
    # print('is_worker_done start')
    jobname = request.args.get('jobname')
    if is_none([jobname]):
        return response_error("clear_worker expects parameters jobname")
    w = workerManager.getWorker(jobname)
    return response_success([not w.getIsProcessRunning()])

@app.route(controller.Client.CleanJobCmd)
def clear_worker():  # type: ignore
    return clear_worker_impl()

def clear_worker_impl() -> Response:
    print('clear_worker start')
    jobname = request.args.get('jobname')
    if is_none([jobname]):
        return response_error("clear_worker expects parameter jobname")
    w = workerManager.getWorker(jobname)
    w.reset()
    w.testAndSetRunning(False)
    return response_success(["ok"])


@app.route(controller.Client.StoreJobCmd)
def storeJob():  # type: ignore
    return storeJobImpl


storeCmd = "./store_experiment_client.sh"
def storeJobImpl() -> Response:
    print('storeJob start')
    jobname = request.args.get('jobname')
    if is_none([jobname]):
        return response_error("storeJob expects parameter jobname")
    for w in workerManager.workers:
        if w.getName() == jobname:
            return response_error("storeJob job still active: " + jobname)

    w = workerManager.storageWorker
    if w.getIsProcessRunning():
        time.sleep(2)
        if w.getIsProcessRunning():
            return response_error("storeJob previous storage worker still running")
    if w.getIsRunning():
        w.reset()
        w.testAndSetRunning(False)
    if not os.path.exists(storeCmd):
        return response_error("command " + storeCmd + " not found from dir " + os.getcwd())
    p = [storeCmd, jobname, "y"]
    w.addNew("storage", p)
    w.setOutput(run_command(p, worker.Worker.getLogfile("storage")))
    return response_success(["ok"])

# helper functions
def is_none(l: List[str]) -> bool:
    for e in l:
        if e == None or len(e) == 0:
            return True
    return False

class ResponseString:
    def __init__(self, msg: str) -> None:
        self.msg = msg


def response_error(msg: str, status_code=500) -> Response:
    res = Response(response=msg, status=status_code, mimetype="text/plain")
    print(msg, flush=True)
    # sys.stdout.flush()
    return res


def toJSON(e: Any) -> str:
    return json.dumps(e, default=lambda o: vars(o))


def response_success(value: Any, root_tag: str=None) -> Response:
    msg = toJSON(value)
    # print(value)
    if root_tag is not None:
        msg = '{"' + root_tag + '": ' + msg + '}'
    res = Response(response=msg, status=200, mimetype="text/json")
    return res

def run_command(command, outfile):
    if not isinstance(command, list):
        raise InvalidUsage("command must be an array")
    print("command is " + " ".join(command))
    print("dir is " + os.getcwd())
    fd = open(outfile, "w", -1)  # negative bufsize means system default
    popen = subprocess.Popen(" ".join(command),   # command needs to be a string
                             stdout=fd, stderr=subprocess.STDOUT, shell=True)  # note shell=True is potentially unsafe (security-wise)
    return (popen, fd)


# app.run(port=5000)
