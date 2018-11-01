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
# run rnn on data (low emb size)
# OMP_NUM_THREADS=2 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -validtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt -testtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$test.txt -nx 50 -nh 50 -lr 0.5 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 90 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 347  -validation_frequency 2*347 -file_prefix save_exp166
# Epoch 652. On validation set: Best (601, 1.001630, 80.1550%)
# saved exp166_e676


# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -nx 50 -nh 50 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp166_e676.zip\$save_exp166_best.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -nx 50 -nh 50 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp166_e676.zip\$save_exp166_best.txt -output_embeddings > train.txt
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt -nx 50 -nh 50 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 90 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp166_e676.zip\$save_exp166_best.txt -output_embeddings > dev.txt
#
# zip -m output/output_embeddings_exp166_e676.zip train.txt dev.txt
# don't add to git (for now), we should make a backup

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

inputfileTrain = "output/output_embeddings_exp166_e676.zip$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/output_embeddings_exp166_e676.zip$dev.txt"
linesDev = confusion_matrix.read_embeddings(inputfileDev, max_line_count=-1)

numberOfClusters = 35
randomSeed = 7485
doShow = True
low = 0
high = 11

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
    print(y1)
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
    plt.title('On dataset $203$. Cheap clustering exp166, emb=$50$')
    plt.plot(x, y1, 'k-', label='Sensitivity')
    plt.plot(x, y2, 'k+', label='Accumulate size')
    # plt.plot(x, y3, 'b-', label='Sensitivity Dev')
    # plt.plot(x, y4, 'b+', label='Accumulate size Dev')
    if show == kutil.SHOW_ALL:
        # plt.plot((low, low), (0, 1), 'k-')
        plt.plot((high - 1, high - 1), (0, 1), 'k:')
    plt.legend()
    plt.savefig('ijcai18_plot_sensitive_sorted_203_166e676.eps')
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

# 83741, 103625
# 17682, 22318
# (+ 17682 22318)40000

kutil.get_base_accuracy(linesC1, "train C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
# if we want to validation score
kutil.get_base_accuracy(lines2C1, "dev C1").report()
kutil.get_base_accuracy(lines2C2, "dev C2").report()

# Accuracy (train C1): 0.9153 (0.9153), f1=0.0006 (+ 2 2 76649 7088)83741
# Accuracy (train C2): 0.7289 (0.5174), f1=0.7124 (+ 34790 12869 40744 15222)103625
# Accuracy (dev C1): 0.9041 (0.9041), f1=0.0000 (+ 0 0 15986 1696)17682
# Accuracy (dev C2): 0.7203 (0.5267), f1=0.6991 (+ 7252 2932 8824 3310)22318

load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
#
# zip -m ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip C1.txt C2.txt 2C1.txt 2C2.txt

# OMP_NUM_THREADS=3 ipython3 functionality/run_model.py -- -nx 50 -nh 50 -L1_reg 0 -L2_reg 0 -retain_probabilities 1 -batch_size 300 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp166_e676.zip\$save_exp166_best.txt
# -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt
# -inputtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt
# -inputtrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$C1.txt
# -inputtrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$C2.txt
# -inputtrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$2C1.txt
# -inputtrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$2C2.txt


# # ## Train set:
# Node count 187366, avg cost 0.080403, avg acc 81.2234%
# Confusion Matrix (tp,fp,tn,fn) 488428 230452 2322246 639590
# # ## dev set:
# Node count 40000, avg cost 0.082881, avg acc 80.1550%
# Confusion Matrix (tp,fp,tn,fn) 103374 51956 490173 140715
# # ## C1 data set:
# Node count 83741, avg cost 0.042569, avg acc 91.5334%
# Confusion Matrix (tp,fp,tn,fn) 6363 33872 1549581 115617
# # ## C2 data set:
# Node count 103625, avg cost 0.113068, avg acc 72.8917%
# Confusion Matrix (tp,fp,tn,fn) 482065 196580 772665 523973
# # ## 2C1 set:
# Node count 17682, avg cost 0.045376, avg acc 90.4083%
# Confusion Matrix (tp,fp,tn,fn) 1748 7228 321376 27434 (=357786)
# # ## 2C2 set:
# Node count 22318, avg cost 0.114201, avg acc 72.0315%
# Confusion Matrix (tp,fp,tn,fn) 101626 44728 168797 113281


#
# Assume we use acc(2C1) == 90.4083
# train using: -lr 0.1 -batch_size 50 -train_report_frequency 1000  -validation_frequency 1000
# report pr 1000 minibatches, i.e. pr 50,000 training instances
#
# exp172 on full data
# exp171 on C2
#
# OMP_NUM_THREADS=2 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$train.txt -validtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$dev.txt -testtrees ../taboo-jan/functionality/203/data_full_random_cleaned.zip\$test.txt -nx 100 -nh 100 -lr 0.1 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 50 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 1000  -validation_frequency 1000 -output_running_model 1000 -file_prefix save_exp172
#
#
# OMP_NUM_THREADS=1 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$C2.txt -validtrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$2C2.txt -testtrees ../taboo-jan/functionality/203/trees_ijcai18_exp166_203.zip\$2C2.txt -nx 100 -nh 100 -lr 0.1 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 50 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 1000  -validation_frequency 1000 -output_running_model 1000 -file_prefix save_exp171

# when models trained, run ijcai18_training_graphs.py


