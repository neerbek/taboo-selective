# -*- coding: utf-8 -*-
"""

Created on October 7, 2017

@author:  neerbek
"""
import unittest

import numpy  # type: ignore
import theano  # type: ignore
import theano.tensor as T  # type: ignore
from theano import pp  # type: ignore
# from theano.ifelse import ifelse

import tests.RunTimer

# in this test I experimented with getting shape information inside a
# theano function.  I can theat the theano compiler and get the shape
# information, but I seem not to be able to use this information in
# exception. I.e. for verifying size of matrices
# But I am able to use the information to create new matrices
#
# Running interpretation: Exceptions are also evaluated when the expression are constructed and at
# that point the dimensions do not uphold the values I expect

class Hmm0:
    def __init__(self):
        self.doDropout = True

class Hmm1:
    def __init__(self, container):
        self.container = container

    def getPrediction(self, x):
        if self.container.doDropout:
            return 3 * x
        else:
            return 4 * x

def f(x):
    o = T.ones_like(x)
    # shape = x.shape
    # shape = (T.sum(o, axis=0), T.sum(o, axis=1))
    # size = shape[0]
    # print("Size is", size)
    return 2 * T.sum(o)

def f2(x):
    o = T.ones_like(x)
    lx = T.sum(o[0, :])
    ly = T.sum(o[:, 0])
    return 2 * lx * ly

def getMatrixShape(x):
    # print("just", type(x) is T.var.TensorVariable)
    lx = T.cast(T.sum(T.ones_like(x[:, 0])), dtype='int32')
    ly = T.cast(T.sum(T.ones_like(x[0, :])), dtype='int32')
    return [lx, ly]

def f3(x):
    shape = getMatrixShape(x)
    return 2 * shape[0] * shape[1]

def f4(x):
    shape = getMatrixShape(x)
    return T.ones(shape=(2 * shape[0], 2 * shape[1]), dtype=theano.config.floatX)

def f5(x, y):
    # xshape = getMatrixShape(x)
    # yshape = getMatrixShape(y)
    # ifelse((xshape[1] != yshape[0]), 0 / 0, 0)
    return T.dot(x, y)

class TmpTest(unittest.TestCase):
    def setUp(self):
        self.timer = tests.RunTimer.Timer()

    def tearDown(self):
        self.timer.report(self, __file__)

    def test_function(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=2 * T.sum(x)
        )
        m = numpy.ones(shape=(20, 19))
        m = m.astype(dtype=theano.config.floatX)
        self.assertEqual(2 * 20 * 19, fx(m))

    def test_function2(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=f(x)
        )
        m = numpy.ones(shape=(20, 19))
        m = m.astype(dtype=theano.config.floatX)
        self.assertEqual(2 * 20 * 19, fx(m))

    def test_function3(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=x.shape
        )
        m = numpy.ones(shape=(20, 19))
        m = m.astype(dtype=theano.config.floatX)
        res = fx(m)
        self.assertEqual(20, res[0])
        self.assertEqual(19, res[1])

    def test_function4(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=f2(x)
        )
        m = numpy.ones(shape=(20, 19))
        m = m.astype(dtype=theano.config.floatX)
        print(pp(f2(x)))
        res = fx(m)
        self.assertEqual(2 * 20 * 19, res)

    def test_function5(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=f3(x)
        )
        m = numpy.ones(shape=(20, 19))
        m = m.astype(dtype=theano.config.floatX)
        res = fx(m)
        self.assertEqual(2 * 20 * 19, res)

    def test_function6(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=f4(x)
        )
        m = numpy.ones(shape=(20, 19))
        m = m.astype(dtype=theano.config.floatX)
        res = fx(m)
        self.assertEqual(2 * 20, res.shape[0])
        self.assertEqual(2 * 19, res.shape[1])

    def test_function7(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        #y = T.matrix('y', dtype=theano.config.floatX)
        # fx = theano.function(
        #     inputs=[x, y],
        #     outputs=f5(x, y)
        # )
        fx2 = theano.function(
            inputs=[x],
            outputs=getMatrixShape(x)
        )
        m1 = 2 * numpy.ones(shape=(20, 7))
        m1 = m1.astype(dtype=theano.config.floatX)
        m2 = 3 * numpy.ones(shape=(7, 3))
        m2 = m2.astype(dtype=theano.config.floatX)
        print("shape1", fx2(m1))
        print("shape2", fx2(m2))
        # res = fx(m1, m2)
        # self.assertEqual(20, res.shape[0])
        # self.assertEqual(3, res.shape[1])
        # self.assertEqual((20 * 3) * 6 * 7, numpy.sum(res))

    def myfunc(self, x):
        if self.beFunky:
            return 3.141529 * x
        else:
            return 4 * x

    def test_funkyness(self):
        self.beFunky = True
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=self.myfunc(x)
        )
        m1 = numpy.ones(shape=(1, 1))
        m1 = m1.astype(dtype=theano.config.floatX)
        self.assertEqual(3.141529, fx(m1))
        self.beFunky = False
        self.assertEqual(3.141529, fx(m1))
        fx = theano.function(
            inputs=[x],
            outputs=self.myfunc(x)
        )
        self.assertEqual(4, fx(m1))

    def test_funkyness2(self):
        hmm0 = Hmm0()
        hmm1 = Hmm1(hmm0)
        x = T.matrix('x', dtype=theano.config.floatX)
        fx = theano.function(
            inputs=[x],
            outputs=hmm1.getPrediction(x)
        )
        m1 = numpy.ones(shape=(1, 1))
        m1 = m1.astype(dtype=theano.config.floatX)
        self.assertEqual(3, fx(m1))
        hmm0.doDropout = False
        self.assertEqual(3, fx(m1))
        fx = theano.function(
            inputs=[x],
            outputs=hmm1.getPrediction(x)
        )
        self.assertEqual(4, fx(m1))


if __name__ == "__main__":
    unittest.main()
