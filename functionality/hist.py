#  -*- coding: utf-8 -*-
"""
Created on Sep 20

@author: neerbek
"""
import pylab  # type: ignore

# https://stackoverflow.com/questions/1721273/howto-bin-series-of-float-values-into-histogram-in-python
def put_in_buckets(data, buckets=10):
    B = buckets - 1
    minv = min(data)
    maxv = max(data)
    norm = maxv - minv
    bincounts = [0 for i in range(buckets)]
    for d in data:
        b = int((d - minv) / norm * B)
        bincounts[b] += 1
    return bincounts

def plot(bincounts):
    pylab.plot(bincounts, 'o')
    pylab.show()

