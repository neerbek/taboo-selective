#  -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:17

@author: neerbek
"""

import sys

import pylab  # type: ignore

import ai_util
import confusion_matrix
import LogFileReader

# import importlib
# importlib.reload(LogFileReader)

max_line_count = -1
inputfile = None
logfile = "log_"
totaltimer = ai_util.Timer("Total time: ")
evaltimer = ai_util.Timer("Eval time: ")
totaltimer.begin()


def syntax():
    print("""syntax: read_logfile.py [-inputfile <file>][-max_line_count <int>]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
# here you can insert manual arglist if needed
# arglist = "read_logfile -inputfile logs/exp151.zip$exp151.log".split(" ")
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2
    if setting == '-inputfile':
        inputfile = arg
    elif setting == '-max_line_count':
        max_line_count = int(arg)
    else:
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

if inputfile == None:
    raise Exception("Need an inputfile")

epochs = LogFileReader.readLogFile(inputfile)

# epochsFull = epochs
# epochsC2 = epochs
# len(epochsFull)
# epochsFull[-1][0].avg_epoch.epoch
# # epoch = 50 * 2086
# for e in epochsFull:
#     e[0].avg_epoch.epoch = int(4.2227 * e[0].avg_epoch.epoch)
# for e in epochsFull:
#     e[0].avg_epoch.epoch = 2 * int(e[0].avg_epoch.epoch / 2)
# len(epochsC2)
# epochsC2[73][0].avg_epoch.epoch
# epochsC2_backup = epochsC2
# epochsC2 = epochsC2_backup[:74]
# # (+ 11323 3677)15000
# for e in epochsC2_backup:
#     e[1].best.acc = e[1].best.acc * (3677.0 / 15000) + 0.9791 * (11323.0 / 15000)
# epoch = 50 * 494

# (/ 2086.0 494) 4.222672064777328

# epochs = epochs[29:]
# t = epochs[:28]
# t.extend(epochs[29:])
# epochs = t
prev = -1
epoch = -1
# epochs = epochs[0:11814]
for i in range(len(epochs.loglines)):
    logline = epochs.loglines[i]
    epoch = logline.epoch
    if prev >= epoch:
        print("Epoch goes back in time index {}. {} vs. {}".format(i, epoch, prev))
        break
    prev = epoch

# x = [e[0].avg_epoch.epoch for e in epochsC2]
# y1 = [e[0].avg_epoch.acc for e in epochsC2]
# y2 = [0 for e in epochsC2]
# prev = 0
# fullIndex = 0
# for i in range(len(x)):
#     xVal = x[i]
#     while epochsFull[fullIndex][0].avg_epoch.epoch != xVal
# # x = [2 * i  for i in range(len(epochs))]
x = [logline.epoch for logline in epochs.loglines]
y1 = [logline.train.nodeAccuracy for logline in epochs.loglines]
y2 = [logline.validationBest.rootAccuracy for logline in epochs.loglines]
y3 = [logline.validation.rootAccuracy for logline in epochs.loglines]

confusion_matrix.new_graph('Epochs', 'Accuracy (percentage)')

pylab.plot(x, y1, '-r', label="train")  # g:
pylab.plot(x, y2, '-k', label="val best")
# pylab.plot(x, y3, '-g', label="val current")
pylab.legend(loc=4)  # loc=4 (lower right)
# pylab.ylim([70, 90])
# pylab.xlim([0, 10000])

print("saving " + logfile + 'acc_epoch.eps')
pylab.savefig(logfile + 'acc_epoch.eps')
# pylab.savefig(logfile[:-4] + 'acc_epoch_w_current_zoom.eps')
# pylab.show()

# ## cost

# x = [e[0].avg_epoch.epoch for e in epochs]
# y1 = [e[0].avg_epoch.cost for e in epochs]
# y2 = [e[1].best.cost for e in epochs]

# confusion_matrix.new_graph('Epochs', 'Cost')

# pylab.plot(x, y1, '-r', label="train")  # g:
# pylab.plot(x, y2, '-k', label="val best")
# pylab.legend(loc=1)  # loc=4 (lower right)
# pylab.ylim([0.06, 0.08])
# pylab.xlim([0, 10000])

# pylab.savefig(logfile + 'cost_epoch_zoom.eps')
# # pylab.show()
