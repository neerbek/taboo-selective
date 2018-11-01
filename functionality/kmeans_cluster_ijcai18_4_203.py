# -*- coding: utf-8 -*-
"""

Created on January 24, 2018

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

# for information type 203
#
#
# run rnn on data (very low emb size)
# OMP_NUM_THREADS=3 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$test.txt -validtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt -testtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$test.txt -nx 50 -nh 20 -lr 0.5 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 90 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 445/5  -validation_frequency 445 -file_prefix save_exp164
#  Epoch 114. On validation set: Best (110, 1.065507, 77.4675%)


# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -nx 50 -nh 20 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp164_epoch480.zip\$save_exp164_best.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -nx 50 -nh 20 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp164_epoch480.zip\$save_exp164_best.txt -output_embeddings > train.txt
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt -nx 50 -nh 20 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp164_epoch480.zip\$save_exp164_best.txt -output_embeddings > dev.txt
#
# zip output_embeddings_exp164_e480.zip train.txt dev.txt
# rm train.txt dev.txt
# mv output_embeddings_exp164_e480.zip output
# don't add to git (for now), we should make a backup

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

inputfileTrain = "output/output_embeddings_exp164_e120.zip$train.txt"
inputfileTrain = "output/output_embeddings_exp164_e480.zip$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/output_embeddings_exp164_e120.zip$dev.txt"
inputfileDev = "output/output_embeddings_exp164_e480.zip$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
low = 3     # 03
high = 22  # 16 not good

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
    plt.plot(x, y1, 'k-', label='Sensitivity')
    plt.plot(x, y2, 'k+', label='Accumulate size')
    # plt.plot(x, y3, 'b-', label='Sensitivity Dev')
    # plt.plot(x, y4, 'b+', label='Accumulate size Dev')
    if show == kutil.SHOW_ALL:
        # plt.plot((low, low), (0, 1), 'k-')
        plt.plot((high, high), (0, 1), 'k:')
    plt.legend()
    plt.savefig('ijcai18_plot_sensitive_sorted_203.eps')
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

# after some iterations (unknown random seed)
# 78442 45824
# 17034 9966 (27000)

kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
# if we want to validation score
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

# don't know if these values are updated!
# Accuracy (train C1): 0.9432 (0.6436), f1=0.9179 (24901 1398 49089 3054)
# Accuracy (train C2): 0.9871 (0.0128), f1=0.9935 (45224 579 8 13)
# Accuracy (dev C1): 0.9304 (0.6318), f1=0.9023 (5470 383 10379 802)
# Accuracy (dev C2): 0.9832 (0.0167), f1=0.9915 (9796 163 3 4)
