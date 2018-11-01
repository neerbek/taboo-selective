# -*- coding: utf-8 -*-
"""

Created on February 6, 2018

@author:  neerbek
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

# for information type 202
#
#
# using exp186 _20000 and _200000 (and maybe _100000)

# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp186.zip\$save_exp186_running_200000.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp186.zip\$save_exp186_running_200000.txt -output_embeddings > train.txt
#
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$dev.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp186.zip\$save_exp186_running_200000.txt -output_embeddings > dev.txt
#
# zip -m output/output_embeddings_exp186_m200K.zip train.txt dev.txt
# don't add to git (for now, 45MB), we should make a backup

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

basefile = "output_embeddings_exp186_m200K.zip"
inputfileTrain = "output/" + basefile + "$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/" + basefile + "$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
low = 10
high = 30  # high = 24 (40%, min 98%), high = 30 (25%, min 99%)

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
    # y1 = kutil.get_cluster_sen_ratios(aTrainFull, linesTrainFull, kmeans, sort_order)
    # y2 = getScaledSizes(aTrainFull, kmeans, sort_order)
    # y3 = kutil.get_cluster_sen_ratios(aDev, linesDev, kmeans, sort_order)
    # y4 = getScaledSizes(aDev, kmeans, sort_order)
    x = kutil.getXRange(show, low, high, numberOfClusters)
    y1 = [y1[i] for i in x]
    y2 = [y2[i] for i in x]
    # y3 = [y3[i] for i in x]
    # y4 = [y4[i] for i in x]
    confusion_matrix.new_graph('Clusters', 'Ratio')
    plt.title('On dataset $202$. Cheap clustering exp186m200K, emb=$100$')
    plt.plot(x, y1, 'k-', label='Sensitivity')
    plt.plot(x, y2, 'k+', label='Accumulate size')
    # plt.plot(x, y3, 'b-', label='Sensitivity Dev')
    # plt.plot(x, y4, 'b+', label='Accumulate size Dev')
    if show == kutil.SHOW_ALL:
        # plt.plot((low, low), (0, 1), 'k-')
        plt.plot((high, high), (0, 1), 'k:')
    plt.legend()
    plt.savefig('ijcai18_plot_sensitive_sorted_202_186m200K_1.eps')
    # plt.savefig('tmp.eps')
    # plt.show() don't call show from an interactive prompt :(
    # https://github.com/matplotlib/matplotlib/issues/8505/

clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:high], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[high:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:high], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[high:], linesDev, aDev, kmeans)
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))

# 89587 34679
# 19469 7531
# (+ 19469 7531)27000

kutil.get_base_accuracy(linesTrainFull, "train").report()
kutil.get_base_accuracy(linesDev, "dev").report()
kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()


# Accuracy (train): 0.9523 (0.4110), f1=0.9586 (68495 1225 49849 4697)
# Accuracy (dev): 0.9450 (0.4047), f1=0.9526 (14918 330 10598 1154)
# Accuracy (train C1): 0.9361 (0.5679), f1=0.9224 (34017 1027 49849 4694)
# Accuracy (train C2): 0.9942 (0.0057), f1=0.9971 (34478 198 0 3)
# Accuracy (dev C1): 0.9267 (0.5584), f1=0.9125 (7443 274 10598 1154)
# Accuracy (dev C2): 0.9926 (0.0074), f1=0.9963 (7475 56 0 0)

load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
#
# zip -m ../taboo-jan/functionality/202/trees_ijcai18_exp186m200K_202.zip C1.txt C2.txt 2C1.txt 2C2.txt

# when models trained, run ijcai18_training_graphs.py
