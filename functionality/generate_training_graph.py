# -*- coding: utf-8 -*-
"""

Created on March 5, 2018

@author:  neerbek
"""

import pylab  # type: ignore
import math

import confusion_matrix
import LogFileReader

import importlib

# exp199
fullfile = "/home/neerbek/jan/phd/DLP/paraphrase/taboo-jan/functionality/logs/exp199.zip$exp199.log"

title = 'On dataset $201$, showing the minibatch cutoffs. exp199'

graphfile = "exp199_standard_training"

file_prefix = "../figures/"

cuts = [50, 100, 150, 200, 250, 500]
colors = ['g:', '-g', 'm:', '-m', 'k:', '-k']

cutoff = 150  # ignore the last part of the graph
# common #####
loglinesFull = LogFileReader.readLogFile(fullfile)
print(len(loglinesFull.loglines))

def testCountMonotone(log, logname):
    for i in range(len(log.loglines)):
        if log.loglines[i].count != (i + 1) * 1000:
            print("mismatch in " + logname + " log on index: ", i)
            break

def testEpochMonotone(log, logname):
    prev = -1
    epoch = -1
    for i in range(len(log.loglines)):
        logline = log.loglines[i]
        epoch = logline.epoch
        if prev > epoch:
            print("Epoch goes back in time in log {}, index {}. {} vs. {}".forloglinesC2.loglinesmat(logname, i, epoch, prev))
            break
        prev = epoch

def convergeToLog(x, y):
    newX = []
    newY = []
    if len(x) != len(y):
        raise Exception("x and y should have same lengths!")
    prev = -1
    for i in range(len(x)):
        lnx = int(math.log(math.log(x[i] + 1)) * 100)
        if lnx > prev:
            newX.append(float(lnx) / 100)
            newY.append(y[i])
    return (newX, newY)

def getAcc(minibatch, log):
    for logline in log.loglines:
        if logline.count == minibatch:
            return logline.validationBest.rootAccuracy
    return 1


testCountMonotone(loglinesFull, "full")
testEpochMonotone(loglinesFull, "full")

x2 = [logline.count for logline in loglinesFull.loglines]
y2 = [logline.validationBest.rootAccuracy for logline in loglinesFull.loglines]

if cutoff != None:
    x2 = x2[1:cutoff]
    y2 = y2[1:cutoff]
# log scale x
# (x2, y2) = convergeToLog(x2, y2)


# plot
print(len(x2), len(y2))
confusion_matrix.new_graph('Visited Minibatches', 'Accuracy')
pylab.title(title)

for i in range(0, len(cuts)):
    cut = cuts[i]
    c = colors[i]
    top = getAcc(cut * 1000, loglinesFull)
    pylab.plot((cut * 1000, cut * 1000), (y2[0], top), c, label="{}K".format(cut))

pylab.plot(x2, y2, '-r', label="Training Accuracy")  # g:
pylab.legend(loc=4)  # loc=4 (lower right)
# pylab.ylim([75, 85])
# pylab.xlim([0, 10000])

print("saving " + file_prefix + graphfile + '_1.eps')
pylab.savefig(file_prefix + graphfile + '_1.eps')

