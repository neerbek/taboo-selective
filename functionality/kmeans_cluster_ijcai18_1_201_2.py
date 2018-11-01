# -*- coding: utf-8 -*-
"""

Created on January 10, 2018

@author:  neerbek

data working for ijcai18, try out 1
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

# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp145.zip\$save_exp145_best.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp145.zip\$save_exp145_best.txt -output_embeddings > train.txt
#
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$dev.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp145.zip\$save_exp145_best.txt -output_embeddings > dev.txt
#
# zip output_embeddings_exp145.zip train.txt dev.txt
# rm train.txt dev.txt
# mv output_embeddings_exp145.zip output

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

inputfileTrain = "output/output_embeddings_exp145.zip$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/output_embeddings_exp145.zip$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
low = 11
high = 27

rng = RandomState(randomSeed)
aTrain = confusion_matrix.get_embedding_matrix(linesTrain, normalize=True)
aTrainFull = confusion_matrix.get_embedding_matrix(linesTrainFull, normalize=True)
aDev = confusion_matrix.get_embedding_matrix(linesDev, normalize=True)

kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
# kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrainFull)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

show = kutil.SHOW_ALL
if doShow:
    # plot
    # y1 = kutil.get_cluster_sen_ratios(aDev, linesDev, kmeans, sort_order)
    # y1 = kutil.get_cluster_sizes(aTrain, kmeans)
    # y1 = [y1[i] for i in sort_order]  # use sort_order order
    y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
    y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
    x = kutil.getXRange(show, low, high, numberOfClusters)
    y1 = [y1[i] for i in x]
    y2 = [y2[i] for i in x]
    confusion_matrix.new_graph('Clusters', 'Sensitive Ratio')
    plt.title('On dataset $201$. Showing the distribution after full classical training')
    plt.plot(x, y1, 'g:', label='Sensitivity')
    plt.plot(x, y2, 'k:', label='Sizes')
    if show == kutil.SHOW_ALL:
        plt.plot((low - 1, low - 1), (0, 1), 'k-')
        # plt.plot((high -1 , high - 1), (0, 1), 'k:')
    plt.legend()
    plt.savefig('ijcai18_plot_201_2.eps')
    # plt.show() don't call show from an interactive prompt :(
    # https://github.com/matplotlib/matplotlib/issues/8505/

clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesDev, aDev, kmeans)
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))

# 44057 60199
# 6214 8786
# (+ 6214 8786)15000

kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

# Accuracy (train C1): 0.9967 (0.9967), f1=0.0265 (2 2 43908 145)
# Accuracy (train C2): 0.9550 (0.7753), f1=0.8931 (11307 491 46184 2217)
# Accuracy (dev C1): 0.9837 (0.9836), f1=0.0194 (1 0 6112 101)
# Accuracy (dev C2): 0.9359 (0.7853), f1=0.8406 (1485 162 6738 401)


load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
#
# zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip C1.txt C2.txt 2C1.txt 2C2.txt

# OMP_NUM_THREADS=3 ipython3 functionality/run_model.py -- -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probabilities 1 -batch_size 300 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp145.zip\$save_exp145_best.txt
# -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt
# -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$dev.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$C1.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$C2.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$2C1.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$2C2.txt


# ## Train set:
# Node count 104256, avg cost 0.025813, avg acc 97.2615%
# Confusion Matrix (tp,fp,tn,fn) 208120 27148 2769443 184565
# ## dev set:
# Node count 15000, avg cost 0.032478, avg acc 95.5733%
# Confusion Matrix (tp,fp,tn,fn) 28242 6726 392462 31185
# ## C1 data set:
# Node count 44057, avg cost 0.004777, avg acc 99.6663%
# Confusion Matrix (tp,fp,tn,fn) 121 8557 1678532 4001
# ## C2 data set:
# Node count 60199, avg cost 0.049562, avg acc 95.5016%
# Confusion Matrix (tp,fp,tn,fn) 207999 18591 1090911 180564
# ## 2C1 set:
# Node count 6214, avg cost 0.011679, avg acc 98.3746%
# Confusion Matrix (tp,fp,tn,fn) 209 2102 232727 3627
# ## 2C2 set:
# Node count 8786, avg cost 0.055045, avg acc 93.5921%
# Confusion Matrix (tp,fp,tn,fn) 28033 4624 159735 27558


#
# Assume we use acc(2C1) == 98.3746
# train using: -lr 0.1 -batch_size 50 -train_report_frequency 1000  -validation_frequency 1000
# report pr 1000 minibatches, i.e. pr 50,000 training instances
# exp173 on C2
# exp174 on full data

# OMP_NUM_THREADS=1 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$C2.txt -validtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$2C2.txt -testtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145_newcut.zip\$2C2.txt -nx 100 -nh 100 -lr 0.1 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 50 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 1000  -validation_frequency 1000 -output_running_model 1000 -file_prefix save_exp173

# OMP_NUM_THREADS=1 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -validtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$dev.txt -testtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$test.txt -nx 100 -nh 100 -lr 0.1 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 50 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 1000  -validation_frequency 1000 -output_running_model 1000 -file_prefix save_exp174

# when models trained, run ijcai18_training_graphs.py

