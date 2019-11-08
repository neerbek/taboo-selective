# -*- coding: utf-8 -*-
"""

Created on November 3, 2017

@author:  neerbek
"""

import numpy  # type: ignore
from numpy.random import RandomState  # type: ignore
import theano  # type: ignore

import ai_util
import confusion_matrix

def addNoise(npArray, rng=RandomState(), noiseFactor=None):
    if noiseFactor is None:
        noiseFactor = 1 / npArray.shape[1]
    tmp = npArray + ((rng.rand(npArray.shape[0], npArray.shape[1]) - 0.5) * noiseFactor)
    tmp = tmp.astype(dtype=theano.config.floatX)
    return tmp

def addBiasColumn(npArray):
    tmp = numpy.ones(shape=(npArray.shape[0], npArray.shape[1] + 1))
    tmp[:, 1:] = npArray
    tmp = tmp.astype(dtype=theano.config.floatX)
    return tmp

def lines2multiclassification(lines, classes):
    yArray = numpy.zeros(shape=(len(lines), len(classes)))
    if len(lines) > 0:
        if not isinstance(lines[0], confusion_matrix.Line):
            raise Exception("Expected a list of Lines", type(lines[0]))
    for i in range(len(lines)):
        classIndex = classes.index(lines[i].ground_truth)  # get index of "ground_truth" in classes, exception if not found
        yArray[i, classIndex] = 1
    yArray = yArray.astype(dtype=theano.config.floatX)
    return yArray

class Timers:
    """Common timers for my command line python scripts.
    Contains a traintimer and totaltimer
    """
    def __init__(self):
        self.totaltimer = ai_util.Timer("Total time: ")
        self.traintimer = ai_util.Timer("Train time: ")
        self.totaltimer.begin()

    def endAndReport(self):
        self.traintimer.end().report()
        self.totaltimer.end().report()
