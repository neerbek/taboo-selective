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

# for information type 202
#
# ~/jan/phd/DLP/paraphrase/taboo-jan/functionality/202
# cat train_full_random.txt dev_full_random.txt test_full_random.txt > data_full_random.txt
# cleanTrees -inputtrees ../taboo-jan/functionality/202/data_full_random.txt -sentenceCutoffLow 5 -sentenceCutoffHigh 200 -outputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.txt
# Count=178266  (* 0.15 178266)
# head -54000 data_full_random_cleaned.txt > tmp.txt
# tail -124266 data_full_random_cleaned.txt > train.txt
# head -27000 tmp.txt > test.txt
# tail -27000 tmp.txt > dev.txt
# wc -l train.txt test.txt dev.txt data_full_random_cleaned.txt
# zip data_full_random_cleaned.zip train.txt dev.txt test.txt
#
# run rnn on data
# OMP_NUM_THREADS=3 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt -validtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$dev.txt -testtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$test.txt -nx 100 -nh 100 -lr 0.5 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 80 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 259  -validation_frequency 2*259 -file_prefix save_exp152
# Epoch 329. On validation set: Best (56, 0.561828, 94.9926%)


# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp152.zip\$save_exp152_best.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp152.zip\$save_exp152_best.txt -output_embeddings > train.txt
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$dev.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp152.zip\$save_exp152_best.txt -output_embeddings > dev.txt
#
# zip output_embeddings_exp152.zip train.txt dev.txt
# rm train.txt dev.txt
# mv output_embeddings_exp152.zip output

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

inputfileTrain = "output/output_embeddings_exp152.zip$train.txt"
linesTrainFull = confusion_matrix.read_embeddings(inputfileTrain, max_line_count=-1)
linesTrain = [linesTrainFull[i] for i in range(60000)]
inputfileDev = "output/output_embeddings_exp152.zip$dev.txt"
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
    plt.savefig('ijcai18_plot_sensitive_sorted_202.eps')
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

# Accuracy (train C1): 0.9432 (0.6436), f1=0.9179 (24901 1398 49089 3054)
# Accuracy (train C2): 0.9871 (0.0128), f1=0.9935 (45224 579 8 13)
# Accuracy (dev C1): 0.9304 (0.6318), f1=0.9023 (5470 383 10379 802)
# Accuracy (dev C2): 0.9832 (0.0167), f1=0.9915 (9796 163 3 4)

load_trees.put_trees("C1.txt", [l.tree for l in linesC1])
load_trees.put_trees("C2.txt", [l.tree for l in linesC2])
load_trees.put_trees("2C1.txt", [l.tree for l in lines2C1])
load_trees.put_trees("2C2.txt", [l.tree for l in lines2C2])
#
# zip ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip C1.txt C2.txt 2C1.txt 2C2.txt
# rm C1.txt C2.txt 2C1.txt 2C2.txt

# OMP_NUM_THREADS=3 ipython3 functionality/run_model.py -- -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probabilities 1 -batch_size 300 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp152.zip\$save_exp152_best.txt
# -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt
# -inputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$dev.txt
# -inputtrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$C1.txt
# -inputtrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$C2.txt
# -inputtrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$2C1.txt
# -inputtrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$2C2.txt


# ## Train set:
# Node count 124266, avg cost 0.038664, avg acc 95.9410%
# Confusion Matrix (tp,fp,tn,fn) 2370413 262585 1082007 133238
# ## dev set:
# Node count 27000, avg cost 0.041342, avg acc 94.9926%
# Confusion Matrix (tp,fp,tn,fn) 518893 59536 230246 31650
# ## C1 data set:
# Node count 78442, avg cost 0.057795, avg acc 94.3245%
# Confusion Matrix (tp,fp,tn,fn) 442139 243889 1079771 74597
# ## C2 data set:
# Node count 45824, avg cost 0.021129, avg acc 98.7081%
# Confusion Matrix (tp,fp,tn,fn) 1928274 18696 2236 58641
# ## 2C1 set:
# Node count 17034, avg cost 0.062400, avg acc 93.0433%
# Confusion Matrix (tp,fp,tn,fn) 97094 54593 229652 18556
# ## 2C2 set:
# Node count 9966, avg cost 0.022222, avg acc 98.3243%
# Confusion Matrix (tp,fp,tn,fn) 421799 4943 594 13094


#
# Assume we use acc(2C2) == 98.3243
# train using: -lr 0.1 -batch_size 50 -train_report_frequency 1000  -validation_frequency 1000
# report pr 1000 minibatches, i.e. pr 50,000 training instances
#
# exp153 on full data
# exp154 on C1
#
# OMP_NUM_THREADS=3 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$train.txt -validtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$dev.txt -testtrees ../taboo-jan/functionality/202/data_full_random_cleaned.zip\$test.txt -nx 100 -nh 100 -lr 0.1 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 50 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 1000  -validation_frequency 1000 -output_running_model 1000 -file_prefix save_exp153
#
#
# OMP_NUM_THREADS=2 ipython3 functionality/train_model.py -- -traintrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$C1.txt -validtrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$2C1.txt -testtrees ../taboo-jan/functionality/202/trees_ijcai18_exp152_202.zip\$2C1.txt -nx 100 -nh 100 -lr 0.1 -L1_reg 0 -L2_reg 0 -n_epochs -1 -retain_probability 1 -batch_size 50 -valid_batch_size 300 -glove_path ../code/glove/ -train_report_frequency 1000  -validation_frequency 1000 -output_running_model 1000 -file_prefix save_exp154

# when models trained, run ijcai18_training_graphs.py


# (defun calc_ratio (acc) (interactive) (/ (round (+ (* acc 100 (/ 17034.0 27000)) (* 98.32 100 (/ 9966.0 27000)))) 100.0) )

# from logfiles
#       | exp153  | exp154  | exp154 full
# 2000  | 79.0926 | 76.5528 | (calc_ratio 76.5528)84.59
# 3000  | 79.0926 | 79.5996 | (calc_ratio 79.5996)86.51
# 4000  | 82.2815 | 82.0359 | (calc_ratio 82.0359)88.05
# 5000  | 86.1074 | 86.9203 | (calc_ratio 86.9203)91.13
# 6000  | 87.6778 | 86.9203 | (calc_ratio 86.9203)91.13
# 7000  | 89.0037 | 86.9203 | (calc_ratio 86.9203)91.13
# 8000  | 89.0037 | 88.8224 | (calc_ratio 88.8224)92.33
# 9000  | 89.0037 | 88.8224 | (calc_ratio 88.8224)92.33
# 10000 | 90.0222 | 88.8224 | (calc_ratio 88.8224)92.33
# 11000 | 90.0222 | 90.6246 | (calc_ratio 90.6246)93.47
# 12000 | 90.0222 | 90.6246 | (calc_ratio 90.6246)93.47
# 13000 | 90.0222 | 90.6246 | (calc_ratio 90.6246)93.47
# 14000 | 91.2556 | 90.7773 | (calc_ratio 90.7773)93.56
# 15000 | 91.2556 | 91.5228 | (calc_ratio 91.5228)94.03
# 16000 | 91.2556 | 91.5228 | (calc_ratio 91.5228)94.03
# 17000 | 91.2556 | 91.7694 | (calc_ratio 91.7694)94.19
# 18000 | 91.6778 | 91.7694 | (calc_ratio 91.7694)94.19
# 19000 | 91.6778 | 92.0805 | (calc_ratio 92.0805)94.38
# 20000 | 91.6778 | 92.0805 | (calc_ratio 92.0805)94.38

