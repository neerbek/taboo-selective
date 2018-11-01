# -*- coding: utf-8 -*-
"""

Created on November 28, 2017

@author:  neerbek

Common classes for clients and servers
"""

import requests
import json

class Job:
    def __init__(self, jobid=None, param=[]):
        self.jobid = jobid
        if not isinstance(param, list):
            raise Exception("Job expects an array as param")
        self.param = param  # array

    def __repr__(self):
        return "<Job jobid:{} param:{}>".format(self.jobid, self.param)

    # static
    def dictToJob(e):
        j = Job()
        if "isCompleted" in e:
            j = RunningJob()
            j.isCompleted = e["isCompleted"]
        j.jobid = e["jobid"]
        j.param = e["param"]
        return j

class RunningJob(Job):
    def __init__(self, jobid=None, param=[], isCompleted=False):
        Job.__init__(self, jobid, param)
        self.isCompleted = isCompleted

    def __repr__(self):
        return "<RunningJob jobid:{} param:{} isCompleted:{}>".format(self.jobid, self.param, self.isCompleted)

class Jobs:
    def demarshal(jsonString):
        jobList = json.loads(jsonString)
        res = []
        for e in jobList:
            res.append(Job.dictToJob(e))
        return res


class Client:
    JobListCmd = "/jobs"
    SubmitJobCmd = "/job"
    CleanJobCmd = "/clear_worker"
    StoreJobCmd = "/storejob"

    def __init__(self, name, ip, port):
        self.name = name
        self.ip = ip
        self.port = port
        self.emptySlots = 0
        self.jobs = []

    def getRequest(self, cmd, params={}):
        url = self.ip + ":{}".format(self.port) + cmd
        res = requests.get(url, params=params)
        print("getRequest:", res.url)
        res = res.text
        return res

    def updateJobList(self):
        res = self.getRequest(Client.JobListCmd)
        res = Jobs.demarshal(res)
        self.jobs = res
        self.emptySlots = res[-1].param[0]
        return res

    def submitJob(self, job):
        res = self.getRequest(Client.SubmitJobCmd, {"jobname": job.jobid, "params": job.param})
        if res != "[\"ok\"]":
            raise Exception("failed to submit job {} for client {} msg={}".format(job, self, res))
        return True

    def cleanJob(self, job):
        res = self.getRequest(Client.CleanJobCmd, {"jobname": job.jobid})
        if res != "[\"ok\"]":
            raise Exception("failed to clean job {} for client {} msg={}".format(job, self, res))
        self.emptySlots += 1
        if len(self.jobs) > 0:
            self.jobs[-1].param = [self.emptySlots]
        return True

    def storeJob(self, job):
        res = self.getRequest(Client.StoreJobCmd, {"jobname": job.jobid})
        if res != "[\"ok\"]":
            raise Exception("failed to store job {} for client {} msg={}".format(job, self, res))
        return True

    def __repr__(self):
        return "<Client name:{} ip:{} port:{}>".format(self.name, self.ip, self.port)

