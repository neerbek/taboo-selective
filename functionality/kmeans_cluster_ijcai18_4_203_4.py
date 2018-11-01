# -*- coding: utf-8 -*-
"""

Created on February 13, 2018

@author:  neerbek

Rerunning on 203, now with pre-training = m200K
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

# for information type 203
#
# using exp184, emb=100

# mb=   0K, 69.3550%
# mb=   3K, 75.1000%
# mb=  10K, 77.5700%
# mb=  50K, 80.3100%
# mb= 100K, 81.2075%
# mb= 200K, 82.0875% (running: 74.3300%) <-- let's try  (turned out this was best)
# mb= 250K, 82.1750% (running: 82.1500%)
# mb= 500K, 82.6725%
# max around 83.3

# we tried with both m200K and m250K
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp184_m665K.zip\$save_exp184_running_200000.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp184_m665K.zip\$save_exp184_running_200000.txt -output_embeddings > train.txt
#
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp184_m665K.zip\$save_exp184_running_200000.txt -output_embeddings > dev.txt
#
# zip -m output/output_embeddings_exp184_m200K.zip train.txt dev.txt
# zip -m output/output_embeddings_exp184_m250K.zip train.txt dev.txt
# don't add to git (for now, because 48MB), we should make a backup

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

basefile = "output_embeddings_exp184_m200K.zip"
basefile = "output_embeddings_exp184_m250K.zip"
inputfileTrain = "output/" + basefile + "$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/" + basefile + "$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
low = 5   # (m100K) 5 (95.5%, 21%) or 6 (94.5%, 25%)
low = 6   # (m200K) (5, 97.8, 17), (6, 97.1%, 23%), (7, 96.5, 25)
#         # dev     (5, 96.2, 17), (6, 95.3, 23), (7, 94.6, 25)
low = 4   # (m250K) (3, 97.1, 14), (4, 96.4, 21), (5, 94.0, 25)
#         # dev     (3, 95.2, 14), (4, 94.4, 21)
high = 21

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
    print("(low, acc, fraction) = ({}, {:.1f}, {:.0f})".format(low, (1 - y1[low - 1]) * 100, y2[low - 1] * 100))
    # y1 = kutil.get_cluster_sen_ratios(aTrainFull, linesTrainFull, kmeans, sort_order)
    # y2 = getScaledSizes(aTrainFull, kmeans, sort_order)
    # y1 = kutil.get_cluster_sen_ratios(aDev, linesDev, kmeans, sort_order)
    # y2 = kutil.getScaledSizes(aDev, kmeans, sort_order)
    x = kutil.getXRange(show, low, high, numberOfClusters)
    y1 = [y1[i] for i in x]
    y2 = [y2[i] for i in x]
    confusion_matrix.new_graph('Clusters', 'Ratio')
    plt.title('On dataset $203$. Cheap clustering exp184 (m200K), $emb=100$')
    plt.plot(x, y1, 'k-', label='Sensitivity')
    plt.plot(x, y2, 'k+', label='Accumulate size')
    if show == kutil.SHOW_ALL:
        plt.plot((low, low), (0, 1), 'k-')
        # plt.plot((high - 1, high - 1), (0, 1), 'k:')
        plt.legend()
        plt.savefig('ijcai18_plot_sensitive_sorted_4_203_4_1.eps')
        # plt.show() don't call show from an interactive prompt :(
        # https://github.com/matplotlib/matplotlib/issues/8505/

# m200K, low = 6
clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesDev, aDev, kmeans)
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))

# 43962 143404
# 9232 30768
# (+ 43962 143404)187366
# (+ 9232 30768)40000

kutil.get_base_accuracy(linesTrainFull, "train").report()
kutil.get_base_accuracy(linesDev, "dev").report()
kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

# Accuracy (train): 0.7645 (0.6952), f1=0.6966 (50652 37681 92583 6450)
# Accuracy (dev): 0.7433 (0.6935), f1=0.6718 (10510 8520 19222 1748)
# Accuracy (train C1): 0.9785 (0.9785), f1=0.0000 (0 1 43018 943)
# Accuracy (train C2): 0.6988 (0.6084), f1=0.7011 (50652 37680 49565 5507)
# Accuracy (dev C1): 0.9645 (0.9644), f1=0.0061 (1 0 8903 328)
# Accuracy (dev C2): 0.6769 (0.6123), f1=0.6789 (10509 8520 10319 1420)
cmC1 = kutil.get_base_accuracy(lines2C1)
cmC2 = kutil.get_base_accuracy(lines2C2)

formula = "(* (/ {}.0 {}) {})"
formulaC1 = formula.format(len(lines2C1), len(linesDev), "{}")  # little hack "{}" means not setted yet
formulaC2 = formula.format(len(lines2C2), len(linesDev), "{}")
print("formula:  (+ {} {})".format(formulaC1.format(cmC1.get_accuracy()), formulaC2.format(cmC2.get_accuracy())))

# (+ (* (/ 9232.0 40000) 0.964471) (* (/ 30768.0 40000) 0.676937))

load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
#
# zip -m ../taboo-jan/functionality/203/trees_ijcai18_exp184_203.zip C1.txt C2.txt 2C1.txt 2C2.txt

# when models trained, run ijcai18_training_graphs.py
