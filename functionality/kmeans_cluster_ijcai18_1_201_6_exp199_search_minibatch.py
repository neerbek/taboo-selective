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

#       best    running
# m25K  0.8842, 0.8842
# m50K  0.9211, 0.9211
# m100K 0.9255, 0.9133
# m150K 0.9361, 0.9282
# m200K 0.9393, 0.9289
# m250K 0.9417, 0.9417
# m300K 0.9417, 0.9360 (same as 250K)
# m400K 0.9450, 0.9200
# m500K 0.9499, 0.9479
# m750K 0.9525, 0.9525

# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp199_800K.zip\$save_exp199_best_250000.txt -output_embeddings -max_tree_count 100 -max_count 100 -max_embedding_count 10000
#
# for realz
# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$train.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp199_800K.zip\$save_exp199_best_250000.txt -output_embeddings > train250K.txt

# OMP_NUM_THREADS=2 ipython3 functionality/run_model_verbose.py -- -inputtrees ../taboo-jan/functionality/201/data_full_random_cleaned.zip\$dev.txt -nx 100 -nh 100 -L1_reg 0 -L2_reg 0 -retain_probability 1 -batch_size 80 -glove_path ../code/glove/ -inputmodel ../taboo-jan/functionality/logs/exp199_800K.zip\$save_exp199_best_250000.txt -output_embeddings > dev.txt
#
# zip -m output/output_embeddings_201_exp199_m25K.zip train.txt dev.txt
# zip -m output/output_embeddings_201_exp199_m50K.zip train.txt dev.txt
# mv train100K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m100K.zip train.txt dev.txt
# zip -m output/output_embeddings_201_exp199_m150K.zip train.txt dev.txt
# mv train200K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m200K.zip train.txt dev.txt
# mv train250K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m250K.zip train.txt dev.txt
# mv train300K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m300K.zip train.txt dev.txt
# mv train400K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m400K.zip train.txt dev.txt
# mv train500K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m500K.zip train.txt dev.txt
# mv train750K.txt train.txt
# zip -m output/output_embeddings_201_exp199_m750K.zip train.txt dev.txt

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()

basefile = "output_embeddings_201_exp199_m750K.zip"
basefile = "output_embeddings_201_exp199_m500K.zip"
basefile = "output_embeddings_201_exp199_m300K.zip"
basefile = "output_embeddings_201_exp199_m250K.zip"
basefile = "output_embeddings_201_exp199_m200K.zip"
basefile = "output_embeddings_201_exp199_m150K.zip"
basefile = "output_embeddings_201_exp199_m100K.zip"
basefile = "output_embeddings_201_exp199_m50K.zip"
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
low = 4
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
print("(low, acc, fraction) = ({}, {:.1f}%, {:.0f}%)".format(low, (1 - y1[low - 1]) * 100, y2[low - 1] * 100))
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

# 50K   #############################################
# (2, 99.3%, 16%), (3, 99.2%, 21%), (4, 98.6%, 27%)

# ijcai18_plot_1_201_6_exp199_4.eps

# (low, acc, fraction) = (3, 99.2%, 21%)
# 22163 82093
# 3096 11904
# Accuracy (train): 0.9318 (0.8689), f1=0.6819 (7623 1065 89520 6048)
# Accuracy (dev): 0.9211 (0.8675), f1=0.6300 (1007 202 12810 981)
# Accuracy (train C1): 0.9928 (0.9928), f1=0.0000 (0 0 22003 160)
# Accuracy (train C2): 0.9153 (0.8354), f1=0.6868 (7623 1065 67517 5888)
# Accuracy (dev C1): 0.9916 (0.9916), f1=0.0000 (0 0 3070 26)
# Accuracy (dev C2): 0.9028 (0.8352), f1=0.6351 (1007 202 9740 955)

# formula:  (+ (* (/ 3096.0 15000) 0.9916020671834626) (* (/ 11904.0 15000) 0.9028057795698925))

# 50K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_50K.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 100K   #############################################
# (3, 99.1%, 20%) (4, 99.0%, 24%) (5, 99.0%, 28%),

# (low, acc, fraction) = (4, 99.0%, 24%)
# 24788 79468
# 3479 11521

# Accuracy (train): 0.9370 (0.8689), f1=0.7128 (8145 1039 89546 5526)
# Accuracy (dev): 0.9255 (0.8675), f1=0.6570 (1070 199 12813 918)
# Accuracy (train C1): 0.9935 (0.9935), f1=0.0000 (0 0 24627 161)
# Accuracy (train C2): 0.9194 (0.8300), f1=0.7178 (8145 1039 64919 5365)
# Accuracy (dev C1): 0.9894 (0.9894), f1=0.0000 (0 0 3442 37)
# Accuracy (dev C2): 0.9063 (0.8307), f1=0.6646 (1070 199 9371 881)

# formula:  (+ (* (/ 3479.0 15000) 0.9893647599885025) (* (/ 11521.0 15000) 0.9062581373144692))

# 100K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_100K.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 150K   #############################################
# (4, 98.9%, 21%), (5, 98.9%, 25%), (6, 98.8%, 27%)

# (low, acc, fraction) = (4, 98.9%, 21%)

# 22360 81896
# 3090 11910
# Accuracy (train): 0.9496 (0.8689), f1=0.7810 (9380 968 89617 4291)
# Accuracy (dev): 0.9361 (0.8675), f1=0.7205 (1236 207 12805 752)
# Accuracy (train C1): 0.9940 (0.9940), f1=0.0000 (0 0 22226 134)
# Accuracy (train C2): 0.9374 (0.8347), f1=0.7854 (9380 968 67391 4157)
# Accuracy (dev C1): 0.9913 (0.9913), f1=0.0000 (0 0 3063 27)
# Accuracy (dev C2): 0.9217 (0.8353), f1=0.7262 (1236 207 9742 725)

# formula:  (+ (* (/ 3090.0 15000) 0.9912621359223301) (* (/ 11910.0 15000) 0.9217464315701092))
# 150K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_150K.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 200K   #############################################
# (4, 99.4%, 19%), (5, 99.0%, 23%), (6, 98.6%, 27%)

# saving ijcai18_plot_1_201_6_exp199_2.eps
# (low, acc, fraction) = (5, 99.0%, 23%)
# 24348 79908
# 3399 11601
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9943 (0.9943), f1=0.0000 (0 0 24208 140)
# Accuracy (train C2): 0.9444 (0.8307), f1=0.8221 (10269 1181 65196 3262)
# Accuracy (dev C1): 0.9894 (0.9894), f1=0.0000 (0 0 3363 36)
# Accuracy (dev C2): 0.9246 (0.8317), f1=0.7580 (1370 293 9356 582)

# formula:  (+ (* (/ 3399.0 15000) 0.9894086496028244) (* (/ 11601.0 15000) 0.9245754676321006))

# 200K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 250K   #############################################
# (4, 99.2%, 16%), (5, 99.1%, 23%), (6, 99.0%, 27%)

# (low, acc, fraction) = (5, 99.1%, 23%)

# 24228 80028
# 3384 11616
# Accuracy (train): 0.9596 (0.8689), f1=0.8315 (10403 948 89637 3268)
# Accuracy (dev): 0.9417 (0.8675), f1=0.7565 (1359 246 12766 629)
# Accuracy (train C1): 0.9934 (0.9934), f1=0.0000 (0 0 24068 160)
# Accuracy (train C2): 0.9493 (0.8312), f1=0.8369 (10403 948 65569 3108)
# Accuracy (dev C1): 0.9902 (0.9902), f1=0.0000 (0 0 3351 33)
# Accuracy (dev C2): 0.9275 (0.8317), f1=0.7635 (1359 246 9415 596)

# formula:  (+ (* (/ 3384.0 15000) 0.9902482269503546) (* (/ 11616.0 15000) 0.9275137741046832))

# 250K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_250K.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 300K is same as 250K

# 500K   #############################################
# (5, 99.5%, 20%), (6, 99.2%, 24%), (7, 99.1%, 27%)

# saving ijcai18_plot_1_201_6_exp199_3.eps

# (low, acc, fraction) = (6, 99.2%, 24%)
# 25008 79248
# 3575 11425
# Accuracy (train): 0.9671 (0.8689), f1=0.8679 (11261 1017 89568 2410)
# Accuracy (dev): 0.9499 (0.8675), f1=0.7993 (1495 258 12754 493)
# Accuracy (train C1): 0.9954 (0.9954), f1=0.0000 (0 0 24894 114)
# Accuracy (train C2): 0.9582 (0.8289), f1=0.8718 (11261 1017 64674 2296)
# Accuracy (dev C1): 0.9883 (0.9885), f1=0.0000 (0 1 3533 41)
# Accuracy (dev C2): 0.9379 (0.8296), f1=0.8083 (1495 257 9221 452)

# formula:  (+ (* (/ 3575.0 15000) 0.9882517482517482) (* (/ 11425.0 15000) 0.9379431072210066))

# 500K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_500K.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 750K   #############################################
# (3, 99.6%, 7%), (5, 99.2%, 19%), (6, 99.2%, 24%),  (7, 99.2%, 32%)

# saving

# (low, acc, fraction) = (6, 99.2%, 24%)
# 25153 79103
# 3611 11389
# Accuracy (train): 0.9690 (0.8689), f1=0.8712 (10918 475 90110 2753)
# Accuracy (dev): 0.9525 (0.8675), f1=0.8013 (1438 163 12849 550)
# Accuracy (train C1): 0.9942 (0.9942), f1=0.0000 (0 0 25008 145)
# Accuracy (train C2): 0.9610 (0.8290), f1=0.8763 (10918 475 65102 2608)
# Accuracy (dev C1): 0.9886 (0.9886), f1=0.0000 (0 0 3570 41)
# Accuracy (dev C2): 0.9410 (0.8290), f1=0.8106 (1438 163 9279 509)

# formula:  (+ (* (/ 3611.0 15000) 0.9886458044862919) (* (/ 11389.0 15000) 0.9409956976029502))

# 750K zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_750K.zip C1.txt C2.txt 2C1.txt 2C2.txt

