# -*- coding: utf-8 -*-
"""

Created on November 11, 2017

@author:  neerbek
"""
import subprocess
from threading import Lock
from typing import List
from typing import Tuple
from typing import IO
from typing import cast
# from threading import Thread
import os

import controller

FileDescriptorType = IO[str]

class ExceptionWithMessage(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.message = msg

class JobStillRunningException(ExceptionWithMessage):
    def __init__(self, msg):
        ExceptionWithMessage.__init__(self, msg)

class NoWorkerException(ExceptionWithMessage):
    def __init__(self, msg):
        ExceptionWithMessage.__init__(self, msg)

class Worker:
    @staticmethod
    def getLogfile(jobname: str) -> str:
        return jobname + ".log"

    def __init__(self) -> None:
        self.protectedIsRunning: bool = False
        self.protectedName: str = None
        self.protectedPopen: subprocess.Popen = None
        self.protectedPopenStdOut: FileDescriptorType = None
        self.protectedCommand: List[str] = None
        self.protectedFile: FileDescriptorType = None
        self.protectedBytesRead: int = 0
        self.protectedLastLog: List[str] = []
        self.mutex: Lock = Lock()

    def getName(self) -> str:
        with self.mutex:
            return self.protectedName

    def addNew(self, name: str, command: List[str]) -> None:
        self.reset()
        with self.mutex:
            self.protectedName = name
            self.protectedCommand = command

    def reset(self) -> None:  # note does not reset isRunning
        with self.mutex:
            if self.protectedPopen is not None:
                if self.protectedPopen.poll() is None:
                    raise JobStillRunningException("job " + self.protectedName + " not completed. poll={}".format(self.protectedPopen.poll()))
                print(type(self.protectedPopenStdOut))
                fd = self.protectedPopenStdOut
                fd.flush()
                os.fsync(cast(int, fd))  # mypy has troubles with file descriptors
                fd.close()
            self.protectedPopen = None
            self.protectedName = None
            self.protectedCommand = None
            if self.protectedFile is not None:
                self.protectedFile.close()
            self.protectedFile = None
            self.protectedBytesRead = 0
            self.protectedLastLog = []

    def setOutput(self, output: Tuple[subprocess.Popen, FileDescriptorType]) -> None:
        with self.mutex:
            self.protectedPopen = output[0]
            self.protectedPopenStdOut = output[1]

    def getIsProcessRunning(self) -> bool:
        with self.mutex:
            if self.protectedPopen is not None:
                if self.protectedPopen.poll() is None:
                    # try:
                    #     self.protectedPopen.wait(timeout=1)  # quicker close
                    # except subprocess.TimeoutExpired:
                    #     pass
                    if self.protectedPopen.poll() is None:
                        return True
        return False

    def getIsRunning(self) -> bool:
        with self.mutex:
            return self.protectedIsRunning

    def testAndSetRunning(self, value=True) -> bool:
        """tests isRunning in a thread safe fashion. If value==True (default) then
        if isRunning is False, then it is set to True and True is returned
        """
        with self.mutex:
            if self.protectedIsRunning != value:
                self.protectedIsRunning = value
                return True
        return False

    def getCurrentLog(self) -> List[str]:
        with self.mutex:
            logfile = Worker.getLogfile(self.protectedName)
            if self.protectedFile is None:
                self.protectedFile = open(logfile, "r")
                self.protectedBytesRead = 0
            res = []
            size = os.path.getsize(logfile)
            if size == self.protectedBytesRead:
                return []
            while size > self.protectedBytesRead:
                line = self.protectedFile.readline()
                self.protectedBytesRead += len(line)
                line = line.rstrip()
                res.append(line)
                if len(res) > 200:
                    res = res[100:]
            self.protectedLastLog = res
            return res

    def getLastLog(self) -> List[str]:
        with self.mutex:
            res = [line for line in self.protectedLastLog]
            return res

class WorkerManager:
    def __init__(self) -> None:
        self.workers: List[Worker] = []
        self.storageWorker: Worker = None

    def initialize(self, numWorkers: int) -> None:
        if len(self.workers) > 0:
            w: Worker
            for w in self.workers:
                w.reset()   # will raise if w is not stopped
                w.testAndSetRunning(False)
            self.storageWorker.reset()
            self.storageWorker.testAndSetRunning(False)
        self.workers = [Worker() for i in range(numWorkers)]
        self.storageWorker = Worker()

    def getWorker(self, workerName: str) -> Worker:
        for worker in self.workers:
            if worker.getName() == workerName:
                return worker
        if workerName == "storage":
            return self.storageWorker
        raise NoWorkerException("worker: " + workerName + " not found")

    def hasReadyWorker(self) -> bool:
        for i in range(len(self.workers)):
            worker = self.workers[i]
            if not worker.getIsRunning():
                return True
        return False

    def findReadyWorker(self) -> Worker:
        for i in range(len(self.workers)):
            worker = self.workers[i]
            if worker.testAndSetRunning():
                return worker
        return None

    def getJobList(self) -> List[controller.Job]:
        freeWorkers = 0
        res: List[controller.Job] = []
        for i in range(len(self.workers)):
            worker = self.workers[i]
            isCompleted = not worker.getIsProcessRunning()
            with worker.mutex:
                if not worker.protectedIsRunning:
                    freeWorkers += 1
                    continue
                # isProcessRunning = False
                # if worker.protectedPopen != None:
                #     isProcessRunning = (worker.protectedPopen.poll() == None)
                res.append(controller.RunningJob(worker.protectedName, worker.protectedCommand, isCompleted))
        worker = self.storageWorker
        isCompleted = not worker.getIsProcessRunning()
        with worker.mutex:
            if worker.protectedIsRunning:
                res.append(controller.RunningJob(worker.protectedName, worker.protectedCommand, isCompleted))
        res.append(controller.Job("Free Workers", param=[freeWorkers]))
        return res

# class ThreadContainer:
#     def __init__(self, name, workerManager):
#         self.name = name
#         self.workerManager = workerManager
#         self.t = None
#         self.mutex = Lock()
#         self.stopSignal = False

#     def do_stop(self):
#         with self.mutex:
#             self.stopSignal = True

#     def doStop(self):
#         with self.mutex:
#             return self.stopSignal

#     def run_internal(self):
#         work = None
#         while not self.doStop():
#             try:
#                 for worker in workerManager.workers:
#                     with worker.mutex:
#                         if worker.protectedIsRunning and worker.protectedOutput != None:
#                             count = 0
#                             for l in worker.protectOutput:
#                                 if l
#                                 worker.protectedCurrentLog.append(l)
#                 work = self.myparent.get_work()
#             except Exception e:
#                 print(self.name + " worker " + worker.getName() + " threw exception. {}".format(e))
#         time.sleep(10)  # TODO: what should the constant be here

#     def start(self):
#         self.t = Thread(target=self.run_internal, args=())
#         self.t.start()
