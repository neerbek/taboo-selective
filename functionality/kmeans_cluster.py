# -*- coding: utf-8 -*-
"""

Created on November 16, 2018

@author:  neerbek

New kmeans_cluster for splitting train/dev set using clusters
Expects input to be a zip file with embeddings saved in train.txt, dev.txt and test.txt
"""
import os

from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

import ai_util
import confusion_matrix
import kmeans_cluster_util as kutil
import similarity.load_trees as load_trees

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

# input variables
basefile = "output/output_embeddings_201_exp199_m200K.zip"   # expect train.txt, dev.txt and test.txt
sample_cutoff = 60000
randomSeed = 7485
doShow = True
numberOfClusters = 35
low = 15
high = 27   # high is only used for showing cuts, data is only split on low
show = kutil.SHOW_ALL
output_plot_file = 'clusters_plot.eps'

# load stuff
inputfileTrain = basefile + "$train.txt"
print("reading: " + inputfileTrain)
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(sample_cutoff)]
inputfileDev = basefile + "$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)
inputfileTest = basefile + "$test.txt"
linesTest = confusion_matrix.read_embeddings(inputfileTest, max_line_count=-1)


rng = RandomState(randomSeed)
aTrain = confusion_matrix.get_embedding_matrix(linesTrain, normalize=True)
aTrainFull = confusion_matrix.get_embedding_matrix(linesTrainFull, normalize=True)
aDev = confusion_matrix.get_embedding_matrix(linesDev, normalize=True)
aTest = confusion_matrix.get_embedding_matrix(linesTest, normalize=True)

# ################
rng = RandomState(randomSeed)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

y1 = getClusterSenRatios(aTrain, treesTrain, kmeans, sort_order)
y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order, accumulate=True)

# ###############
if doShow:
    fy1 = ["{:.4f}".format(1 - y) for y in y1]
    fy2 = ["{:.2f}".format(100 * y) for y in y2]
    print("acc", fy1)
    print("sizes", fy2)

    # acc ['0.9948', '0.9881', '0.9825', '0.9824', '0.9815', '0.9809', '0.9771', '0.9702', '0.9659', '0.9515', '0.7826', '0.7101', '0.1319', '0.1194', '0.1064']
    # sizes ['11.92', '27.84', '33.66', '40.58', '46.71', '51.34', '58.62', '67.91', '76.07', '81.94', '85.96', '89.12', '91.91', '94.55', '100.00']
    x = kutil.getXRange(show, low, high, numberOfClusters)
    y1 = [y1[i] for i in x]   # clones or cuts y1 (y2) depending on value of x
    y2 = [y2[i] for i in x]

    confusion_matrix.new_graph('Clusters', 'Sensitive Ratio')
    # plt.title('... your titel here ...')
    plt.plot(x, y1, 'k:', label='Train sensitivity')
    plt.plot(x, y2, 'g:', label='Train size')
    if show == kutil.SHOW_ALL:
        plt.plot((low, low), (0, 1), 'k-')
        plt.plot((high, high), (0, 1), 'k-')
    plt.legend()
    plt.savefig(output_plot_file)
    # plt.show() don't call show from an interactive prompt :(
    # https://github.com/matplotlib/matplotlib/issues/8505/

# ###############
clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesDev, aDev, kmeans)
(lines3C1, a3C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTest, aTest, kmeans)
(lines3C2, a3C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTest, aTest, kmeans)
print("(numberOfClusters, low, acc, acc[c+1], fraction) = ({}, {}, {:.4f}%, {:.4f}%, {:.4f}%)".format(numberOfClusters, low, (1 - y1[low - 1]) * 100, (1 - y1[low]) * 100, y2[low - 1] * 100))
print("(devAcc, devFraction) = ({:.4f}%, {:.4f}%)".format((1 - y1dev[low - 1]) * 100, y2[low - 1] * 100))
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))
print(len(lines3C1), len(lines3C2))

kutil.get_base_accuracy(linesTrainFull, "train").report()
kutil.get_base_accuracy(linesDev, "dev").report()
kutil.get_base_accuracy(linesTest, "test").report()
kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()
kutil.get_base_accuracy(lines3C1, "test C1").report()
kutil.get_base_accuracy(lines3C2, "test C2").report()

cmC1 = kutil.get_base_accuracy(lines2C1)
cmC2 = kutil.get_base_accuracy(lines2C2)

formula = "(* (/ {}.0 {}) {})"   # for emacs calc
formulaC1 = formula.format(len(lines2C1), len(linesDev), "{}")  # little hack "{}" means not setted yet
formulaC2 = formula.format(len(lines2C2), len(linesDev), "{}")
print("formula:  (+ {} {})".format(formulaC1.format(cmC1.get_accuracy()), formulaC2.format(cmC2.get_accuracy())))

load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
load_trees.put_trees("3C1.txt", [l.tree for l in lines3C1])
load_trees.put_trees("3C2.txt", [l.tree for l in lines3C2])

# zip -m <output-zip> C1.txt C2.txt 2C1.txt 2C2.txt 3C1.txt 3C2.txt
