# -*- coding: utf-8 -*-
"""

Created on February 12, 2018

@author:  neerbek

Search for best fraction, based on exp199_200K
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

# using 200K
# ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K.zip

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()
basefile = "output_embeddings_201_exp199_m200K.zip"
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
low = 15
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
    print("(low, acc, fraction) = ({}, {:.1f}%, {:.1f}%)".format(low, (1 - y1[low - 1]) * 100, y2[low - 1] * 100))
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

# (low, acc, fraction) = (1, 99.7%, 3.6%)
# 3859 100397
# 499 14501
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9966 (0.9966), f1=0.0000 (0 0 3846 13)
# Accuracy (train C2): 0.9545 (0.8640), f1=0.8180 (10269 1181 85558 3389)
# Accuracy (dev C1): 0.9940 (0.9940), f1=0.0000 (0 0 496 3)
# Accuracy (dev C2): 0.9374 (0.8631), f1=0.7511 (1370 293 12223 615)
# formula:  (+ (* (/ 499.0 15000) 0.9939879759519038) (* (/ 14501.0 15000) 0.937383628715261))

# 5% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_5p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 10%   #############################################
#

# (low, acc, fraction) = (3, 99.4%, 11.9%)
# 12531 91725
# 1696 13304
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9951 (0.9951), f1=0.0000 (0 0 12470 61)
# Accuracy (train C2): 0.9507 (0.8516), f1=0.8196 (10269 1181 76934 3341)
# Accuracy (dev C1): 0.9906 (0.9906), f1=0.0000 (0 0 1680 16)
# Accuracy (dev C2): 0.9327 (0.8518), f1=0.7538 (1370 293 11039 602)
# formula:  (+ (* (/ 1696.0 15000) 0.9905660377358491) (* (/ 13304.0 15000) 0.9327269993986771))

# 10% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_10p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 20%   #############################################
#

# (low, acc, fraction) = (4, 99.4%, 19.3%)
# 20209 84047
# 2803 12197
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9948 (0.9948), f1=0.0000 (0 0 20104 105)
# Accuracy (train C2): 0.9467 (0.8386), f1=0.8210 (10269 1181 69300 3297)
# Accuracy (dev C1): 0.9907 (0.9907), f1=0.0000 (0 0 2777 26)
# Accuracy (dev C2): 0.9274 (0.8391), f1=0.7559 (1370 293 9942 592)
# formula:  (+ (* (/ 2803.0 15000) 0.9907242240456654) (* (/ 12197.0 15000) 0.9274411740591949))

# 20% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_20p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 25%   #############################################
#

# (low, acc, fraction) = (6, 98.6%, 26.9%)
# 28189 76067
# 3890 11110
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9933 (0.9933), f1=0.0000 (0 0 28001 188)
# Accuracy (train C2): 0.9422 (0.8227), f1=0.8237 (10269 1181 61403 3214)
# Accuracy (dev C1): 0.9884 (0.9884), f1=0.0000 (0 0 3845 45)
# Accuracy (dev C2): 0.9221 (0.8251), f1=0.7598 (1370 293 8874 573)
# formula:  (+ (* (/ 3890.0 15000) 0.9884318766066839) (* (/ 11110.0 15000) 0.922052205220522))

# 25% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_25p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 30%   #############################################
#

# (low, acc, fraction) = (7, 98.6%, 31.0%)
# 32516 71740
# 4529 10471
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9923 (0.9923), f1=0.0000 (0 0 32265 251)
# Accuracy (train C2): 0.9396 (0.8129), f1=0.8258 (10269 1181 57139 3151)
# Accuracy (dev C1): 0.9870 (0.9870), f1=0.0000 (0 0 4470 59)
# Accuracy (dev C2): 0.9186 (0.8158), f1=0.7628 (1370 293 8249 559)
# formula:  (+ (* (/ 4529.0 15000) 0.9869728416869066) (* (/ 10471.0 15000) 0.91863241333206))

# 30% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_30p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 40%   #############################################
#

# (low, acc, fraction) = (9, 98.4%, 39.9%)
# 41840 62416
# 5882 9118
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9904 (0.9904), f1=0.0000 (0 2 41437 401)
# Accuracy (train C2): 0.9330 (0.7874), f1=0.8309 (10269 1179 47967 3001)
# Accuracy (dev C1): 0.9864 (0.9864), f1=0.0000 (0 0 5802 80)
# Accuracy (dev C2): 0.9089 (0.7907), f1=0.7673 (1370 293 6917 538)
# formula:  (+ (* (/ 5882.0 15000) 0.9863991839510371) (* (/ 9118.0 15000) 0.9088615924544856))

# 40% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_40p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 50%   #############################################
#

# (low, acc, fraction) = (12, 98.0%, 50.6%)
# 53064 51192
# 7432 7568
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9880 (0.9881), f1=0.0000 (0 2 52428 634)
# Accuracy (train C2): 0.9229 (0.7453), f1=0.8388 (10269 1179 36976 2768)
# Accuracy (dev C1): 0.9820 (0.9820), f1=0.0000 (0 0 7298 134)
# Accuracy (dev C2): 0.8973 (0.7550), f1=0.7791 (1370 293 5421 484)
# formula:  (+ (* (/ 7432.0 15000) 0.9819698600645855) (* (/ 7568.0 15000) 0.897330866807611))

# 50% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_50p.zip C1.txt C2.txt 2C1.txt 2C2.txt

# 60%   #############################################
#

# (low, acc, fraction) = (15, 97.2%, 59.9%)
# 62638 41618
# 8775 6225
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9866 (0.9865), f1=0.0277 (12 8 61784 834)
# Accuracy (train C2): 0.9101 (0.6918), f1=0.8458 (10257 1173 27620 2568)
# Accuracy (dev C1): 0.9802 (0.9799), f1=0.0333 (3 1 8598 173)
# Accuracy (dev C2): 0.8816 (0.7089), f1=0.7877 (1367 292 4121 445)
# formula:  (+ (* (/ 8775.0 15000) 0.9801709401709402) (* (/ 6225.0 15000) 0.8816064257028112))

# 60% zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_60p.zip C1.txt C2.txt 2C1.txt 2C2.txt

