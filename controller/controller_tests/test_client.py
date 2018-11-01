# -*- coding: utf-8 -*-
"""

Created on November 11, 2017

@author:  neerbek
"""

import os
import unittest
import requests
import json
import subprocess
import time
from flask import Response

# os.chdir("..")
# os.chdir("../../../taboo-core")

import tests.RunTimer as RunTimer
import controller
import controller_client_app
controller_client_app.initializeModule(numWorkers=1)


class Args:
    def __init__(self):
        self.args = {}

    def get(self, key):
        if key in self.args:
            return self.args[key]
        return None

    def add(self, key, value):
        if key in self.args:
            self.args[key] = value
        if not isinstance(self.args[key], list):
            self.args[key] = [self.args[key]]
        self.args[key].append(value)

    def getlist(self, key):
        return self.args[key]


class Request:
    def __init__(self):
        self.args = Args()

# self = ClientTest()
class ClientTest(unittest.TestCase):
    def setUp(self):
        self.timer = RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_hello_world(self):
        controller_client_app.initializeModule(numWorkers=1)
        controller_client_app.request = Request()
        res = controller_client_app.hello_world()
        self.assertTrue(isinstance(res, Response))
        returnVal = res.get_data().decode('utf-8')
        self.assertTrue(isinstance(returnVal, str), "expected string response " + str(type(returnVal)))
        print(returnVal)

    def test_path(self):
        p = os.getcwd()
        d = os.path.basename(p)
        self.assertEqual(d, "controller")

    def test_has_empty_worker(self):
        controller_client_app.initializeModule(numWorkers=1)
        controller_client_app.request = Request()
        res = controller_client_app.has_empty_worker()
        res = res.get_data().decode('utf-8')
        self.assertTrue(isinstance(res, str), "expected string response")
        self.assertEqual("[true]", res)
        d = json.loads(res)
        self.assertTrue(isinstance(d, list), "expected list result")
        self.assertEqual(True, d[0])

    def test_get_joblist(self):
        controller_client_app.initializeModule(numWorkers=1)
        controller_client_app.request = Request()
        res = controller_client_app.getJobList()
        res = res.get_data().decode('utf-8')
        self.assertTrue(isinstance(res, str))
        res = controller.Jobs.demarshal(res)
        self.assertTrue(isinstance(res, list))
        self.assertEqual(1, len(res))
        self.assertTrue(isinstance(res[0], controller.Job))
        self.assertEqual(1, res[0].param[0])
        self.assertEqual("Free Workers", res[0].jobid)

    def test_run_job(self):
        controller_client_app.initializeModule(numWorkers=1)
        jobname = "test_exp001"
        controller_client_app.runJobCmd = "controller_tests/resources/hello_world.py"
        controller_client_app.request = Request()
        controller_client_app.request.args.args['params'] = "this is a long list of arguments"
        controller_client_app.request.args.args['jobname'] = jobname
        res = controller_client_app.run_job_cmd()
        self.assertTrue(isinstance(res, Response))
        res = res.get_data().decode('utf-8')
        self.assertEqual("[\"ok\"]", res)
        worker = controller_client_app.workerManager.getWorker(jobname)
        worker.protectedPopen.wait(timeout=1)
        # it = iter(worker.protectedPopen.stdout.readline, b'')
        print("printing output")
        # for l in it:
        #     print(l)
        res = controller_client_app.clear_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[\"ok\"]", res)

    def test_run_job_wait(self):
        controller_client_app.initializeModule(numWorkers=1)
        jobname = "test_exp002"
        self.startServer(jobname, 5001)
        controller_client_app.request = Request()
        res = controller_client_app.has_empty_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[false]", res)
        controller_client_app.request = Request()
        res = controller_client_app.getJobList()
        res = res.get_data().decode('utf-8')
        self.assertTrue(isinstance(res, str))
        res = controller.Jobs.demarshal(res)
        self.assertTrue(isinstance(res, list))
        self.assertEqual(2, len(res))
        self.assertTrue(isinstance(res[0], controller.RunningJob))
        self.assertEqual("controller_tests/resources/wait_for_input.py", res[0].param[2])
        self.assertEqual(jobname, res[0].jobid)
        self.assertEqual(False, res[0].isCompleted)
        self.assertEqual(0, res[1].param[0])
        self.assertEqual("Free Workers", res[1].jobid)

        self.waitReady([jobname], [5001])

        res = requests.get("http://127.0.0.1:5001/stop").status_code
        self.assertEqual(200, res)
        self.waitStop([jobname], [5000])
        controller_client_app.request = Request()
        res = controller_client_app.getJobList()
        res = res.get_data().decode('utf-8')
        res = controller.Jobs.demarshal(res)
        self.assertTrue(isinstance(res, list))
        self.assertEqual(2, len(res))
        self.assertTrue(isinstance(res[0], controller.RunningJob))
        self.assertEqual("controller_tests/resources/wait_for_input.py", res[0].param[2])
        self.assertEqual(jobname, res[0].jobid)
        self.assertEqual(True, res[0].isCompleted)
        self.assertEqual(0, res[1].param[0])
        self.assertEqual("Free Workers", res[1].jobid)

        controller_client_app.request = Request()
        controller_client_app.request.args.args['jobname'] = jobname
        res = controller_client_app.clear_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[\"ok\"]", res)

    def startServer(self, jobname, port):
        controller_client_app.runJobCmd = "controller_tests/resources/wait_for_input.py"
        controller_client_app.request = Request()
        controller_client_app.request.args.args['params'] = ["-port", "{}".format(port)]
        controller_client_app.request.args.args['jobname'] = jobname
        res = controller_client_app.run_job_cmd()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[\"ok\"]", res)

    def waitReady(self, jobnames, ports):
        numWorkers = len(jobnames)
        responses = [None for i in range(numWorkers)]
        start = time.time()
        count = 0
        while time.time() < start + 3:
            isDone = True
            for i in range(numWorkers):
                if responses[i] == 200:
                    continue
                isDone = False
                url = "http://127.0.0.1:{}/ready".format(ports[i])
                try:
                    responses[i] = requests.get(url).status_code
                except Exception:
                    pass
            count += 1
            if isDone:
                break
            time.sleep(0.02)
        for i in range(numWorkers):
            print(str(responses[i]))
        for i in range(numWorkers):
            if responses[i] != 200:
                raise Exception("Worker " + jobnames[i] + " is not ready")
        print("time waiting: {:.4f}. Count {}".format(time.time() - start, count))

    def waitStop(self, jobnames, ports):
        numWorkers = len(jobnames)
        responses = [None for i in range(numWorkers)]
        start = time.time()
        count = 0
        while time.time() < start + 10:
            isDone = True
            # i = 0
            # i += 1
            for i in range(numWorkers):
                if responses[i] == "[true]":
                    continue
                isDone = False
                controller_client_app.request = Request()
                controller_client_app.request.args.args['jobname'] = jobnames[i]
                res = controller_client_app.is_worker_done()
                res = res.get_data().decode('utf-8')
                # print(res)
                responses[i] = res
            count += 1
            if isDone:
                break
            time.sleep(0.02)
        for i in range(numWorkers):
            if responses[i] != "[true]":
                raise Exception("Worker " + jobnames[i] + " is not stopped. " + str(responses[i]) + " isDone=" + str(isDone))
        print("time waiting: {:.4f}. Count {}".format(time.time() - start, count))

    def test_run_job_multiple(self):
        numWorkers = 3
        controller_client_app.initializeModule(numWorkers=numWorkers)
        jobnames = ["test_multi001", "test_multi002", "test_multi003"]
        ports = [5001, 5002, 5003]
        responses = [None, None, None]
        for i in range(numWorkers):
            responses[i] = self.startServer(jobnames[i], ports[i])
        controller_client_app.request = Request()
        res = controller_client_app.has_empty_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[false]", res)
        self.waitReady(jobnames, ports)

        for i in range(numWorkers):
            url = "http://127.0.0.1:{}/stop".format(ports[i])
            res = requests.get(url).status_code
            self.assertEqual(200, res)
        self.waitStop(jobnames, ports)

        for i in range(numWorkers):
            controller_client_app.request = Request()
            controller_client_app.request.args.args['jobname'] = jobnames[i]
            res = controller_client_app.clear_worker()
            res = res.get_data().decode('utf-8')
            self.assertEqual("[\"ok\"]", res)

    def test_run_job_multiple_adding_new(self):
        numWorkers = 3
        controller_client_app.initializeModule(numWorkers=numWorkers)
        jobnames = ["test_multi001", "test_multi002", "test_multi003"]
        ports = [5001, 5002, 5003]
        responses = [None, None, None]
        for i in range(numWorkers):
            responses[i] = self.startServer(jobnames[i], ports[i])
        controller_client_app.request = Request()
        res = controller_client_app.has_empty_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[false]", res)
        self.waitReady(jobnames, ports)
        jobname = jobnames[1]
        port = ports[1]
        url = "http://127.0.0.1:{}/stop".format(port)
        res = requests.get(url).status_code
        self.assertEqual(200, res)
        self.waitStop([jobname], [port])
        controller_client_app.request = Request()
        controller_client_app.request.args.args['jobname'] = jobname
        res = controller_client_app.clear_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[\"ok\"]", res)
        controller_client_app.request = Request()
        res = controller_client_app.has_empty_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[true]", res)
        jobnames[1] = "test_multi004"
        ports[1] = 5002
        responses[1] = self.startServer(jobnames[1], ports[1])
        controller_client_app.request = Request()
        res = controller_client_app.has_empty_worker()
        res = res.get_data().decode('utf-8')
        self.assertEqual("[false]", res)
        self.waitReady(jobnames, ports)

        for i in range(numWorkers):
            url = "http://127.0.0.1:{}/stop".format(ports[i])
            res = requests.get(url).status_code
            self.assertEqual(200, res)
        self.waitStop(jobnames, ports)

        for i in range(numWorkers):
            controller_client_app.request = Request()
            controller_client_app.request.args.args['jobname'] = jobnames[i]
            res = controller_client_app.clear_worker()
            res = res.get_data().decode('utf-8')
            self.assertEqual("[\"ok\"]", res)

    def test_popen(self):
        pipe = subprocess.Popen(["echo jan en to tre"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pipe.wait()
        it = iter(pipe.stdout.readline, b'')
        for i, l in enumerate(it):
            pipe.wait()
            print(l)
        pipe.stdout.close()
        pipe.stderr.close()

