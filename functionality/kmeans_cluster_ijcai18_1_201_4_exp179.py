# -*- coding: utf-8 -*-
"""

Created on January 10, 2018

@author:  neerbek

data working for ijcai18, try out 3

using different models from different counts of minibatches over exp176
Note exp176 didn't get that good acc (acc=90.6 vs 95.9 (best ever))
"""
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

# running on full data set, another option could be to run on C2...
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp179_e569.zip\$save_exp179_running_100000.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp179_e569.zip\$save_exp179_running_100000.txt -output_embeddings > train.txt
#
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$dev.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp179_e569.zip\$save_exp179_running_100000.txt -output_embeddings > dev.txt
#
# zip -m output/output_embeddings_exp179_m100000.zip train.txt dev.txt

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

inputfileTrain = "output/output_embeddings_exp179_m100000.zip$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/output_embeddings_exp179_m100000.zip$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
low = 16  # about 65%% of data, acc=96.63
low = 7   # about 36%% of data, acc=97.5
low = 5   # about 28%% of data, acc=97.6
high = 35
rng = RandomState(randomSeed)
aTrain = confusion_matrix.get_embedding_matrix(linesTrain, normalize=True)
aTrainFull = confusion_matrix.get_embedding_matrix(linesTrainFull, normalize=True)
aDev = confusion_matrix.get_embedding_matrix(linesDev, normalize=True)

kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
# kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrainFull)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

show = kutil.SHOW_LOW
if doShow:
    # plot
    # y1 = kutil.get_cluster_sen_ratios(aDev, linesDev, kmeans, sort_order)
    # y2 = kutil.getScaledSizes(aDev, kmeans, sort_order)
    y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
    y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
    x = kutil.getXRange(show, low, high, numberOfClusters)
    y1 = [y1[i] for i in x]
    y2 = [y2[i] for i in x]
    confusion_matrix.new_graph('Clusters', 'Sensitive Ratio')
    plt.title('On dataset $201$, using emb from exp179, after $100K$ minibatches (+ $600K$ from earlier).\nShowing the distribution after full classical training')
    plt.plot(x, y1, 'g:', label='Sensitivity')
    plt.plot(x, y2, 'k:', label='Sizes')
    if show == kutil.SHOW_ALL:
        plt.plot((low - 1, low - 1), (0, 1), 'k-')
        # plt.plot((high -1 , high - 1), (0, 1), 'k:')
    plt.legend()
    plt.savefig('ijcai18_plot_201_4.eps')
    # plt.show() don't call show from an interactive prompt :(
    # https://github.com/matplotlib/matplotlib/issues/8505/

clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesDev, aDev, kmeans)
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))

# low = 7 ############################################################
#
# 38449 65807
# 5412 9588

kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

# Accuracy (train C1): 0.9855 (0.9855), f1=0.0000 (0 0 37892 557)
# Accuracy (train C2): 0.8912 (0.8007), f1=0.6341 (6204 249 52444 6910)
# Accuracy (dev C1): 0.9754 (0.9754), f1=0.0000 (0 0 5279 133)
# Accuracy (dev C2): 0.8834 (0.8065), f1=0.5871 (795 58 7675 1060)


load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
#
# zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp179_newcut.zip C1.txt C2.txt 2C1.txt 2C2.txt

#
# Assume we use acc(2C1) == 97.70
# exp179 on C2
# exp174 on full data

# when models trained, run ijcai18_training_graphs.py

