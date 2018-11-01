# -*- coding: utf-8 -*-
"""

Created on January 18, 2018

@author:  neerbek

File for plotting training acc vs mini batches
"""

import pylab  # type: ignore
import math

import confusion_matrix
import LogFileReader

import importlib
importlib.reload(LogFileReader)

convertC2Accuracy = None

# 201
fullfile = "logs/exp150.zip$exp150.log"
c2file = "logs/exp151.zip$exp151.log"
logfile = "ijcai_faster_training_201"
offset = 200000

# 201 attempt 2
fullfile = "logs/exp174.zip$exp174.log"
c2file = "logs/exp151.zip$exp151.log"
logfile = "ijcai_faster_training_201_exp174_vs_exp151"
offset = 200000

title = 'On dataset $201$, working on lower cluster. Showing convergence of the two approaches'

def convertC2Accuracy201(localAcc):
    return (localAcc * 3677.0 / 15000 + (15000.0 - 3677.0) / 15000 * 97.91)


convertC2Accuracy = convertC2Accuracy201

# 202
fullfile = "logs/exp153.zip$exp153.log"
c2file = "logs/exp154.zip$exp154.log"
logfile = "ijcai_faster_training_202"
offset = 200000


title = 'On dataset $202$, working on lower cluster. Showing convergence of the two approaches'

def convertC2Accuracy202(localAcc):
    return (localAcc * 17034.0 / 27000 + (27000.0 - 17034.0) / 27000 * 98.32)


convertC2Accuracy = convertC2Accuracy202

# 203
fullfile = "logs/exp184_5M.zip$exp184.log"
c2file = "logs/exp195_10M.zip$exp195.log"
logfile = "ijcai_faster_training_203"
offset = 200000  # m for clustering


title = 'On dataset $203$, working on lower cluster. Showing convergence of the two approaches'

# (+ (* (/ 9232.0 40000) 96.4471) (* (/ 30768.0 40000) 79.8914))83.71245556
def convertC2Accuracy203(localAcc):
    return (localAcc * 30768.0 / 40000 + (40000.0 - 30768.0) / 40000 * 96.4471)


convertC2Accuracy = convertC2Accuracy203

# common #####
loglinesFull = LogFileReader.readLogFile(fullfile)
loglinesC2 = LogFileReader.readLogFile(c2file)
print(len(loglinesFull.loglines), len(loglinesC2.loglines))
# > 199, 122
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


testCountMonotone(loglinesFull, "full")
testCountMonotone(loglinesC2, "C2")
testEpochMonotone(loglinesFull, "full")
testEpochMonotone(loglinesC2, "C2")

x1 = [i * 1000 for i in range(1, int(offset / 1000))]
x1.extend([logline.count + offset for logline in loglinesC2.loglines])
y1 = [0 for i in range(1, int(offset / 1000))]
y1.extend([convertC2Accuracy(logline.validationBest.rootAccuracy) for logline in loglinesC2.loglines])
x2 = [logline.count for logline in loglinesFull.loglines]
y2 = [logline.validationBest.rootAccuracy for logline in loglinesFull.loglines]

# same lenths
m = min(len(loglinesFull.loglines), len(loglinesC2.loglines))
x1 = x1[:m]
y1 = y1[:m]
x2 = x2[:m]
y2 = y2[:m]

# remove first
x1 = x1[1:]
y1 = y1[1:]
x2 = x2[1:]
y2 = y2[1:]

# remove some
x1 = x1[:4500]
y1 = y1[:4500]
x2 = x2[:4500]
y2 = y2[:4500]

# log scale x
(x1, y1) = convergeToLog(x1, y1)
(x2, y2) = convergeToLog(x2, y2)

# plot
print(len(x1), len(y1), len(x2), len(y2))
confusion_matrix.new_graph('Visited Minibatches', 'Accuracy')
pylab.title(title)

pylab.plot(x1, y1, '-r', label="Our approach")  # g:
pylab.plot(x2, y2, '-k', label="Standard approach")
# pylab.plot(x, y3, '-g', label="val current")
pylab.legend(loc=4)  # loc=4 (lower right)
pylab.ylim([75, 85])
# pylab.xlim([0, 10000])

print("saving " + logfile + '_2.eps')
pylab.savefig(logfile + '_2.eps')

# mv ijcai_faster_training_1.eps ~/jan/phd/Articles/ijcai2018_svn/figures
#
# mv ijcai_faster_training_202_1.eps ~/jan/phd/Articles/ijcai2018_svn/figures
# this is nice, clustered approach is better than classical approach at all times
# minor drawback, we assume access to the clusters. However IDEA: we can generate the clusteres using embeddings of low size. I.e then the number of calculations are much lower.
# still need outside dataset
# revisit 201 where we saw no improvement, try adding more training data


# 203 ################

# remove some
x1 = [i * 1000 for i in range(1, int(offset / 1000))]
x1.extend([logline.count + offset for logline in loglinesC2.loglines])
y1 = [0 for i in range(1, int(offset / 1000))]
y1.extend([convertC2Accuracy(logline.validationBest.rootAccuracy) for logline in loglinesC2.loglines])
x2 = [logline.count for logline in loglinesFull.loglines]
y2 = [logline.validationBest.rootAccuracy for logline in loglinesFull.loglines]

# same lenths
x1 = x1[:4500]
y1 = y1[:4500]
x2 = x2[:4500]
y2 = y2[:4500]

confusion_matrix.new_graph('Visited Minibatches', 'Accuracy')
pylab.title(title)

pylab.plot(x1, y1, '-r', label="Our approach")  # g:
pylab.plot(x2, y2, '-k', label="Standard approach")
pylab.legend(loc=4)  # loc=4 (lower right)
pylab.ylim([75, 85])

print("saving " + logfile + '_1.eps')
# saving ijcai_faster_training_203_1.eps
pylab.savefig(logfile + '_1.eps')

# shows nicely our improved acc and faster training. Our approach lies above classical approach after initial training

# 203 full #######
x1 = [i * 1000 for i in range(1, int(offset / 1000))]
x1.extend([logline.count + offset for logline in loglinesC2.loglines])
y1 = [0 for i in range(1, int(offset / 1000))]
y1.extend([convertC2Accuracy(logline.validationBest.rootAccuracy) for logline in loglinesC2.loglines])
x2 = [logline.count for logline in loglinesFull.loglines]
y2 = [logline.validationBest.rootAccuracy for logline in loglinesFull.loglines]

confusion_matrix.new_graph('Visited Minibatches', 'Accuracy')
pylab.title(title)

pylab.plot(x1, y1, '-r', label="Our approach")  # g:
pylab.plot(x2, y2, '-k', label="Standard approach")
pylab.legend(loc=4)  # loc=4 (lower right)
pylab.ylim([75, 85])

print("saving " + logfile + '_2.eps')
# saving ijcai_faster_training_203_2.eps
pylab.savefig(logfile + '_2.eps')

# We have continued our approach training for much longer (both
# approaches are still (220218) running) than classical. Here we show
# both. Classical actually have a good improvement around 5000Km
# interesting to see where it goes from here. Right now it kind of
# looks bad (but not really, we are good)
