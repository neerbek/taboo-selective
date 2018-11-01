# -*- coding: utf-8 -*-
"""

Created on February 12, 2018

@author:  neerbek

New cut based of exp194 (full 201, emb=100, lr=0.4, bs=50)
"""
# -*- coding: utf-8 -*-
import os
os.chdir("../../taboo-core")
from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

import ai_util
import confusion_matrix
import kmeans_cluster_util as kutil
import similarity.load_trees as load_trees

# import pylab  # type: ignore
import matplotlib.pyplot as plt
import importlib
# importlib.reload(kutil)
importlib.reload(confusion_matrix)

# using 100K
# ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K.zip

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()
basefile = "output_embeddings_201_exp199_m100K.zip"
inputfileTrain = "output/" + basefile + "$train.txt"
print("reading: " + inputfileTrain)
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/" + basefile + "$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
# from exp194: # (low, acc, fraction) = (5, 99.1%, 23%)
low = 11
high = 27

rng = RandomState(randomSeed)
aTrain = confusion_matrix.get_embedding_matrix(linesTrain, normalize=True)
aTrainFull = confusion_matrix.get_embedding_matrix(linesTrainFull, normalize=True)
aDev = confusion_matrix.get_embedding_matrix(linesDev, normalize=True)

kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
# kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrainFull)
# kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aDev)
# sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrainFull, linesTrainFull, kmeans)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)


show = kutil.SHOW_ALL
if doShow:
    # plot
    y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
    y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
    print("(low, acc, fraction) = ({}, {:.1f}%, {:.0f}%)".format(low, (1 - y1[low - 1]) * 100, y2[low - 1] * 100))
    # y1 = kutil.get_cluster_sen_ratios(aTrainFull, linesTrainFull, kmeans, sort_order)
    # y2 = getScaledSizes(aTrainFull, kmeans, sort_order)
    # y1 = kutil.get_cluster_sen_ratios(aDev, linesDev, kmeans, sort_order)
    # y2 = kutil.getScaledSizes(aDev, kmeans, sort_order)
    x = kutil.getXRange(show, low, high, numberOfClusters)
    y1 = [y1[i] for i in x]
    y2 = [y2[i] for i in x]
    confusion_matrix.new_graph('Clusters', 'Ratio')
    plt.title('On dataset $201$, using emb from exp194, after $50K$ minibatches')
    plt.plot(x, y1, 'g:', label='Sensitivity')
    plt.plot(x, y2, 'k:', label='Accumulate size')
    if show == kutil.SHOW_ALL:
        plt.plot((low, low), (0, 1), 'k-')
        # plt.plot((high -1 , high - 1), (0, 1), 'k:')
    plt.legend()
    outfile = 'ijcai18_plot_1_201_6_exp199_4.eps'
    print("saving " + outfile)
    plt.savefig(outfile)
    # plt.show() don't call show from an interactive prompt :(
    # https://github.com/matplotlib/matplotlib/issues/8505/

clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesDev, aDev, kmeans)
print("(low, acc, fraction) = ({}, {:.1f}%, {:.1f}%)".format(low, (1 - y1[low - 1]) * 100, y2[low - 1] * 100))
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))

kutil.get_base_accuracy(linesTrainFull, "train").report()
kutil.get_base_accuracy(linesDev, "dev").report()
kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

cmC1 = kutil.get_base_accuracy(lines2C1)
cmC2 = kutil.get_base_accuracy(lines2C2)

formula = "(* (/ {}.0 {}) {})"
formulaC1 = formula.format(len(lines2C1), len(linesDev), "{}")  # little hack "{}" means not setted yet
formulaC2 = formula.format(len(lines2C2), len(linesDev), "{}")
print("formula:  (+ {} {})".format(formulaC1.format(cmC1.get_accuracy()), formulaC2.format(cmC2.get_accuracy())))

load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])

# 5%   #############################################
#

# (low, acc, fraction) = (1, 99.6%, 6.1%)
# 6360 97896
# 856 14144
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9964 (0.9964), f1=0.0000 (0 0 6337 23)
# Accuracy (train C2): 0.9332 (0.8606), f1=0.7135 (8145 1039 83209 5503)
# Accuracy (dev C1): 0.9977 (0.9977), f1=0.0000 (0 0 854 2)
# Accuracy (dev C2): 0.9212 (0.8596), f1=0.6575 (1070 199 11959 916)
# formula:  (+ (* (/ 856.0 15000) 0.9976635514018691) (* (/ 14144.0 15000) 0.9211679864253394))

# 5% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_5p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 15%   #############################################
#

# (low, acc, fraction) = (2, 99.4%, 16.2%)
# 16903 87353
# 2400 12600
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9951 (0.9951), f1=0.0000 (0 0 16820 83)
# Accuracy (train C2): 0.9258 (0.8444), f1=0.7154 (8145 1039 72726 5443)
# Accuracy (dev C1): 0.9933 (0.9933), f1=0.0000 (0 0 2384 16)
# Accuracy (dev C2): 0.9126 (0.8435), f1=0.6603 (1070 199 10429 902)
# formula:  (+ (* (/ 2400.0 15000) 0.9933333333333333) (* (/ 12600.0 15000) 0.9126190476190477))

# 15% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_15p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 20%   #############################################
#

# (low, acc, fraction) = (3, 99.1%, 20.5%)
# 21377 82879
# 3024 11976
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9942 (0.9942), f1=0.0000 (0 0 21254 123)
# Accuracy (train C2): 0.9223 (0.8365), f1=0.7166 (8145 1039 68292 5403)
# Accuracy (dev C1): 0.9907 (0.9907), f1=0.0000 (0 0 2996 28)
# Accuracy (dev C2): 0.9091 (0.8363), f1=0.6627 (1070 199 9817 890)
# formula:  (+ (* (/ 3024.0 15000) 0.9907407407407407) (* (/ 11976.0 15000) 0.9090681362725451))

# 20% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_20p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 25%   #############################################
#

# (low, acc, fraction) = (4, 99.0%, 23.7%)
# 24788 79468
# 3479 11521
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9935 (0.9935), f1=0.0000 (0 0 24627 161)
# Accuracy (train C2): 0.9194 (0.8300), f1=0.7178 (8145 1039 64919 5365)
# Accuracy (dev C1): 0.9894 (0.9894), f1=0.0000 (0 0 3442 37)
# Accuracy (dev C2): 0.9063 (0.8307), f1=0.6646 (1070 199 9371 881)
# formula:  (+ (* (/ 3479.0 15000) 0.9893647599885025) (* (/ 11521.0 15000) 0.9062581373144692))

# 25% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_25p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 30%   #############################################
#

# (low, acc, fraction) = (6, 98.4%, 31.9%)
# 33305 70951
# 4650 10350
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9916 (0.9917), f1=0.0000 (0 1 33026 278)
# Accuracy (train C2): 0.9114 (0.8112), f1=0.7216 (8145 1038 56520 5248)
# Accuracy (dev C1): 0.9871 (0.9871), f1=0.0000 (0 0 4590 60)
# Accuracy (dev C2): 0.8979 (0.8137), f1=0.6694 (1070 199 8223 858)
# formula:  (+ (* (/ 4650.0 15000) 0.9870967741935484) (* (/ 10350.0 15000) 0.8978743961352657))

# 30% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_30p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 40%   #############################################
#

# (low, acc, fraction) = (8, 98.2%, 40.9%)
# 42771 61485
# 6015 8985
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9898 (0.9898), f1=0.0000 (0 1 42334 436)
# Accuracy (train C2): 0.9003 (0.7847), f1=0.7266 (8145 1038 47212 5090)
# Accuracy (dev C1): 0.9854 (0.9854), f1=0.0000 (0 0 5927 88)
# Accuracy (dev C2): 0.8855 (0.7885), f1=0.6753 (1070 199 6886 830)
# formula:  (+ (* (/ 6015.0 15000) 0.9853699085619285) (* (/ 8985.0 15000) 0.8854757929883138))

# 40% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_40p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 50%   #############################################
#

# (low, acc, fraction) = (11, 97.5%, 52.5%)
# 54819 49437
# 7700 7300
# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9872 (0.9872), f1=0.0000 (0 1 54118 700)
# Accuracy (train C2): 0.8814 (0.7376), f1=0.7353 (8145 1038 35428 4826)
# Accuracy (dev C1): 0.9821 (0.9821), f1=0.0000 (0 0 7562 138)
# Accuracy (dev C2): 0.8659 (0.7466), f1=0.6861 (1070 199 5251 780)
# formula:  (+ (* (/ 7700.0 15000) 0.9820779220779221) (* (/ 7300.0 15000) 0.8658904109589041))

# 50% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K_50p.zip C1.txt C2.txt 2C1.txt 2C2.txt

