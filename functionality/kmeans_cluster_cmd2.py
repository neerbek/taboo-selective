# -*- coding: utf-8 -*-
"""

Created on October 3, 2017

@author:  neerbek
"""
import sys
from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

import ai_util
import rnn_model.rnn
import rnn_model.learn
import rnn_model.FlatTrainer
import confusion_matrix
import kmeans_cluster_util as kutil
import similarity.load_trees as load_trees

import pylab  # type: ignore

# import importlib
# importlib.reload(kutil)
inputfile = "../../taboo-core/output_embeddings.txt"

numberOfClusters = 35
low = 16
high = 27
doShow = False
targetClass = 2
randomSeed = 7485
learnRate = 0.5
momentum = 0.0

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()


def syntax():
    print("""syntax: kmeans_cluster_cmd2.py
-inputfile <filename> | -numberOfClusters <int> | -low <int> | -high <int> |
-doShow | -targetClass <int> | -retain_probability <float> | -batchSize <int> |
-randomSeed <int> | [-h | --help | -?]

-inputfile is a list of final sentence embeddings in the format of run_model_verbose.py
-numberOfClusters - how many clusters to generate with kmeans
-low is low cutoff in clustering
-high is high cutoff in clustering
-doShow - show a graph of sensitivity scores for generated clusters
-targetClass (0=all data, 1=lower group, 2=middle group, 3=high group)

-randomSeed initialize the random number generator

kmeans_cluster_cmd2 fits a kmeans model to train set and applies this (predicts) the clustering on the second dataset (dev or test)
""")
    sys.exit()


arglist = sys.argv
# arglist = "kmeans_cluster_cmd2.py -inputfile ../../taboo-core/output_embeddings.txt -numberOfClusters 35 -low 16 -high 27 -randomSeed 37624".split(" ")
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == '-inputfile':
        inputfile = arg
    elif setting == '-numberOfClusters':
        numberOfClusters = int(arg)
    elif setting == '-low':
        low = int(arg)
    elif setting == '-high':
        high = int(arg)
    elif setting == '-targetClass':
        targetClass = int(arg)
    elif setting == '-randomSeed':
        randomSeed = int(arg)
    else:
        # expected option with no argument
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        elif setting == '-doShow':
            doShow = True
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i


lines = confusion_matrix.read_embeddings(inputfile, max_line_count=-1)

rng = RandomState(randomSeed)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, [], rng)
# (lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
confusion_matrix.verify_matrix_normalized(a)
confusion_matrix.verify_matrix_normalized(a2)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)

if doShow:
    # plot
    y1 = kutil.get_cluster_sen_ratios(a, lines, kmeans, sort_order)
    x = range(len(y1))
    confusion_matrix.new_graph('Clusters', 'Sensitive Ratio')
    pylab.plot(x, y1, 'g:', label='Train $1$')
    pylab.plot((low, low), (0, 1), 'k-')
    pylab.plot((high, high), (0, 1), 'k-')
    pylab.legend()
    print("close graph window to continue...")
    pylab.show()

# cut [:16] + [16:28] + [28:]
# cut [:14] + [14:29] + [29:]
# cut [14:27]
clusterIdRange = range(len(sort_order))
clusterIds = [clusterIdRange[i] for i in sort_order]
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], lines, a, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:high], lines, a, kmeans)
(linesC3, aC3) = kutil.get_sentences_from_clusters(clusterIds[high:], lines, a, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], lines2, a2, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:high], lines2, a2, kmeans)
(lines2C3, a2C3) = kutil.get_sentences_from_clusters(clusterIds[high:], lines2, a2, kmeans)
print(len(linesC1), len(linesC2), len(linesC3))
print(len(lines2C1), len(lines2C2), len(lines2C3))
kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(linesC3, "train C3").report()
# if we want to validation score
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()
kutil.get_base_accuracy(lines2C3, "dev C3").report()

lineList = [linesC1, linesC2, linesC3, lines2C1, lines2C2, lines2C3]
files = ["output/kmeans_embeddingsC1.txt", "output/kmeans_embeddingsC2.txt", "output/kmeans_embeddingsC3.txt", "output/kmeans_embeddings2C1.txt", "output/kmeans_embeddings2C2.txt", "output/kmeans_embeddings2C3.txt"]
for i in range(len(files)):
    f = files[i]
    l = lineList[i]
    tmp = confusion_matrix.read_embeddings(f)
    if len(tmp) != len(l):
        raise Exception("for {} lengths differ. {} vs. {}".format(f, len(tmp), len(l)))
    for j in range(len(l)):
        if not l[j].equals(tmp[j]):
            raise Exception("contents differ [{}]. {} vs. {}".format(j, load_trees.output(l[j].tree), load_trees.output(tmp[j].tree)))
print("all lists are equal")
