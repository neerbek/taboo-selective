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
low = 20
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
        plt.plot((low, low), (0, 1), 'k-')
        # plt.plot((high, high), (0, 1), 'k:')
    plt.legend()
    plt.savefig('ijcai18_plot_201_1.eps')
    # plt.show() don't call show from an interactive prompt :(
    # https://github.com/matplotlib/matplotlib/issues/8505/

clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low + 1], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low + 1:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low + 1], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low + 1:], linesDev, aDev, kmeans)
print(len(linesC1), len(linesC2))
print(len(lines2C1), len(lines2C2))

# 79557 24699
# 11323 3677
kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
# if we want to validation score
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

# Accuracy (train C1): 0.9919 (0.9919), f1=0.0092 (+ 3 2 78909 643)
# Accuracy (train C2): 0.9105 (0.4727), f1=0.9110 (+ 11306 491 11183 1719)
# Accuracy (dev C1): 0.9792 (0.9791), f1=0.0084 (+ 1 0 11086 236)
# Accuracy (dev C2): 0.8836 (0.5238), f1=0.8740 (+ 1485 162 1764 266)


load_trees.put_trees("kmeans_treesC1_full_cleaned.txt", [l.tree for l in linesC1])
load_trees.put_trees("kmeans_treesC2_full_cleaned.txt", [l.tree for l in linesC2])
load_trees.put_trees("kmeans_trees2C1_full_cleaned.txt", [l.tree for l in lines2C1])
load_trees.put_trees("kmeans_trees2C2_full_cleaned.txt", [l.tree for l in lines2C2])
#
# mv kmeans_treesC1_full_cleaned.txt C1.txt
# mv kmeans_treesC2_full_cleaned.txt C2.txt
# mv kmeans_trees2C2_full_cleaned.txt 2C2.txt
# mv kmeans_trees2C1_full_cleaned.txt 2C1.txt
# zip trees_ijcai18_exp145.zip C1.txt C2.txt 2C1.txt 2C2.txt
# rm C1.txt C2.txt 2C1.txt 2C2.txt
# mv trees_ijcai18_exp145.zip ../taboo-jan/functionality/201

# OMP_NUM_THREADS=3 ipython3 functionality/run_model.py -- -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probabilities 1 -batch_size 300 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp145.zip\$save_exp145_best.txt -inputtrees kmeans_trees2C2_full_cleaned.txt
# -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt
# -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$dev.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145.zip\$C1.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145.zip\$C2.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145.zip\$2C1.txt
# -inputtrees ../taboo-jan/functionality/201/trees_ijcai18_exp145.zip\$2C2.txt


# ## Train set:
# total accuracy 93.3617 % (87.6873 %) cost 0.025813, root acc 97.2615 % (86.8871 %)
# Confusion Matrix (tp,fp,tn,fn) 208120 27148 2769443 184565
# ## dev set:
# total accuracy 91.7336 % (87.0421 %) cost 0.032478, root acc 95.5733 % (86.7467 %)
# Confusion Matrix (tp,fp,tn,fn) 28242 6726 392462 31185
# ## C1 data set:
# total accuracy 98.6307 % (99.2105 %) cost 0.007427, root acc 99.1893 % (99.1880 %)
# Confusion Matrix (tp,fp,tn,fn) 884 16097 2586910 19829
# ## C2 data set:
# total accuracy 68.9178 % (34.2290 %) cost 0.111113, root acc 91.0523 % (47.2651 %)
# Confusion Matrix (tp,fp,tn,fn) 207236 11051 182533 164736
# ## 2C1 set:
# total accuracy 97.1332 % (97.9875 %) cost 0.013632, root acc 97.9157 % (97.9069 %)
# Confusion Matrix (tp,fp,tn,fn) 482 3677 362769 7044
# ## 2C2 set:
# total accuracy 67.8768 % (38.6825 %) cost 0.115743, root acc 88.3601 % (52.3797 %)
# Confusion Matrix (tp,fp,tn,fn) 27760 3049 29693 24141


# ./start_screen.sh exp149 (3141)
# OMP_NUM_THREADS=3 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/201/ijcai18_exp145.zip\$C2.txt -validtrees ../taboo-jan/functionality/201/ijcai18_exp145.zip\$2C2.txt -testtrees ../taboo-jan/functionality/201/ijcai18_exp145.zip\$2C2.txt -nx 100 -nh 100 -lr 0.5 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 80 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 103  -validation_frequency 3*103 -file_prefix save_exp149

# 150118 00:00 Epoch 10526. On train set : Node count 565556, avg cost nan, avg acc 34.2290%^M
# 150118 00:00 Epoch 10526. On validation set: Best (823, 2.627258, 86.0756%). Current:  total accuracy 38.6825 % (38.6825 %) cost nan, root acc 52.3797 % (52.3797 %)^M

# seems we cannot improve on dataset by only considering a subset of data. Need to criple initial model

# does this model converge faster (than original model)?

#
#
#
# Assume we use acc(2C1) == 97.9157
# train using: -lr 0.1 -batch_size 50 -train_report_frequency 1000  -validation_frequency 1000
# report pr 1000 minibatches, i.e. pr 50,000 training instances
# exp150 on full data
# exp151 on C2

# when models trained, run ijcai18_training_graphs.py

# from logfiles
#       | exp150  | exp151  | exp151 full
# 1000  | 86.7467 | 53.2227 | (+ (* 53.2227 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))86.95
# 2000  | 86.7467 | 53.2227 | (+ (* 53.2227 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))86.95
# 3000  | 86.7467 | 53.2227 | (+ (* 53.2227 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))86.95
# 4000  | 86.7467 | 67.0112 | (+ (* 67.0112 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))90.34
# 5000  | 86.7467 | 73.6742 | (+ (* 73.6742 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))91.97
# 6000  | 86.7467 | 73.6742 | (+ (* 73.6742 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))91.97
# 7000  | 86.7467 | 73.6742 | (+ (* 73.6742 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))91.97
# 8000  | 86.8267 | 74.0821 | (+ (* 74.0821 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.07
# 9000  | 87.0667 | 74.0821 | (+ (* 74.0821 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.07
# 10000 | 87.0667 | 74.1637 | (+ (* 74.1637 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.09
# 11000 | 87.0667 | 76.4482 | (+ (* 76.4482 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.65
# 12000 | 88.2133 | 76.4482 | (+ (* 76.4482 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.65
# 13000 | 88.2133 | 76.4482 | (+ (* 76.4482 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.65
# 14000 | 88.2133 | 76.4482 | (+ (* 76.4482 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))92.65
# 15000 | 88.2133 | 80.8540 | (+ (* 80.8540 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))93.73
# 16000 | 88.8333 | 80.8540 | (+ (* 80.8540 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))93.73
# 17000 | 88.8333 | 80.8540 | (+ (* 80.8540 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))93.73
# 18000 | 89.5267 | 80.8540 | (+ (* 80.8540 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))93.73
# 19000 | 89.5267 | 80.8540 | (+ (* 80.8540 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))93.73
# 20000 | 90.1600 | 80.8540 | (+ (* 80.8540 (/ 3677.0 15000)) (* 97.91 (/ 11323.0 15000)))93.73

