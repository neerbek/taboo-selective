# -*- coding: utf-8 -*-
"""

Created on October 23, 2017

@author:  neerbek
"""

import unittest

import numpy  # type: ignore
from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

import tests.RunTimer

def createData(rng=RandomState(42)):
    nInputMaxRange = 10
    nExamples = 500
    nIn = 10
    xVal = rng.randint(
        nInputMaxRange, size=(nExamples, nIn))
    # x_val = self.x_val.astype(dtype=theano.config.floatX)
    return xVal

def createData2(rng=RandomState(42)):
    nInputMaxRange = 10000
    nExamples = 500
    nIn = 10
    xVal = numpy.sqrt(rng.randint(
        nInputMaxRange, size=(nExamples, nIn)))
    # x_val = self.x_val.astype(dtype=theano.config.floatX)
    return xVal

def getMessage(msg1, msg2):
    if msg2 is None:
        return msg1
    return msg2

class KMeansTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def assertArrayEqual(self, expected, value, msg=None):
        if len(expected) != len(value):
            raise Exception(getMessage("length differs: {} vs. {}".format(len(expected), len(value)), msg))
        for i in range(len(expected)):
            if expected[i] != value[i]:
                raise Exception(getMessage("values differ in array pos={}, {} vs. {}".format(i, len(expected), len(value)), msg))

    def assertArrayClose(self, expected, value, atol=0.0001, msg=None):
        if len(expected) != len(value):
            raise Exception(getMessage("length differs: {} vs. {}".format(len(expected), len(value)), msg))
        for i in range(len(expected)):
            if not numpy.isclose(expected[i], value[i], atol=atol):
                raise Exception(getMessage("values differ in array pos={}, {} vs. {}".format(i, len(expected), len(value)), msg))

    def test_random(self):
        """test that random is the same across machines"""
        xVal = createData(rng=RandomState(42))
        self.assertArrayEqual([6, 3, 7, 4, 6, 9, 2, 6, 7, 4], numpy.asarray(xVal[0:1, :]).reshape(xVal.shape[1]))
        xVal = createData(rng=RandomState(38739))
        self.assertArrayEqual([0, 6, 0, 3, 2, 6, 7, 1, 3, 6], numpy.asarray(xVal[0:1, :]).reshape(xVal.shape[1]))

    def test_random2(self):
        """test that random float is the same across machines"""
        xVal = createData2(rng=RandomState(42))
        self.assertArrayClose([85.26429499, 29.3257566, 73.41661937, 72.04859471, 75.72318007,
                               79.15175298, 21.58703314, 66.52818951, 74.6860094, 91.22499657], numpy.asarray(xVal[0:1, :]).reshape(xVal.shape[1]))
        xVal = createData2(rng=RandomState(38739))
        self.assertArrayClose([93.88823142, 69.61321714, 82.44998484, 94.64671151, 19.26136028,
                               46.61544808, 65.2533524, 95.115719, 80.71554993, 71.42128534], numpy.asarray(xVal[0:1, :]).reshape(xVal.shape[1]))

    def test_kmeans(self):
        rng = RandomState(42)
        xVal = createData(rng)
        kmeans = KMeans(n_clusters=20, random_state=rng).fit(xVal)
        res = kmeans.predict(xVal)
        # print(res[0:20])
        self.assertArrayEqual([2, 13, 8, 1, 8, 17, 11, 3, 14, 7, 8, 9, 6, 7, 4, 14, 9, 17, 0, 13], res[0:20])

    def test_kmeans2(self):
        rng = RandomState(42)
        xVal = createData2(rng)
        kmeans = KMeans(n_clusters=20, random_state=rng).fit(xVal)
        res = kmeans.predict(xVal)
        print(res[0:20])
        self.assertArrayEqual([10, 15, 3, 13, 1, 15, 6, 8, 8, 12, 5, 14, 5, 13, 12, 12, 18, 15, 2, 18], res[0:20])

