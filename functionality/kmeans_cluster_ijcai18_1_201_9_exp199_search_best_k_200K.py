# -*- coding: utf-8 -*-
"""

Created on March 9, 2018

@author:  neerbek

Search for k values
"""
import os
os.chdir("../../taboo-core")
from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore

import ai_util
import confusion_matrix
import kmeans_cluster_util as kutil
import similarity.load_trees as load_trees

import importlib
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

numberOfClusters = 15
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
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

# ################
numberOfClusters = 15
rng = RandomState(randomSeed)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
fy1 = ["{:.4f}".format(1 - y) for y in y1]
fy2 = ["{:.2f}".format(100 * y) for y in y2]
print("acc", fy1)
print("sizes", fy2)

# acc ['0.9948', '0.9881', '0.9825', '0.9824', '0.9815', '0.9809', '0.9771', '0.9702', '0.9659', '0.9515', '0.7826', '0.7101', '0.1319', '0.1194', '0.1064']
# sizes ['11.92', '27.84', '33.66', '40.58', '46.71', '51.34', '58.62', '67.91', '76.07', '81.94', '85.96', '89.12', '91.91', '94.55', '100.00']

# ################
numberOfClusters = 35
rng = RandomState(randomSeed)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
fy1 = ["{:.4f}".format(1 - y) for y in y1]
fy2 = ["{:.2f}".format(100 * y) for y in y2]
print("acc", fy1)
print("sizes", fy2)

# acc ['0.9972', '0.9958', '0.9945', '0.9942', '0.9903', '0.9863', '0.9859', '0.9858', '0.9838', '0.9810', '0.9805', '0.9802', '0.9790', '0.9789', '0.9724', '0.9718', '0.9705', '0.9704', '0.9691', '0.9685', '0.9627', '0.9532', '0.9258', '0.9234', '0.9196', '0.8709', '0.8393', '0.7721', '0.6288', '0.2350', '0.2106', '0.1557', '0.0982', '0.0881', '0.0743']
# sizes ['3.62', '8.00', '11.91', '19.34', '23.27', '26.92', '31.04', '34.32', '39.89', '45.24', '48.92', '50.61', '53.62', '56.31', '59.93', '63.77', '65.98', '68.62', '70.67', '72.90', '74.19', '77.33', '78.56', '79.67', '82.08', '82.99', '84.81', '87.62', '88.60', '90.19', '91.29', '92.68', '96.38', '98.03', '100.00']

# ################
numberOfClusters = 70
rng = RandomState(randomSeed)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)

y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
fy1 = ["{:.4f}".format(1 - y) for y in y1]
fy2 = ["{:.2f}".format(100 * y) for y in y2]
print("acc", fy1)
print("sizes", fy2)

# acc ['0.9967', '0.9957', '0.9956', '0.9955', '0.9948', '0.9943', '0.9937', '0.9929', '0.9927', '0.9924', '0.9924', '0.9922', '0.9918', '0.9912', '0.9910', '0.9899', '0.9899', '0.9854', '0.9854', '0.9848', '0.9844', '0.9838', '0.9831', '0.9819', '0.9804', '0.9791', '0.9788', '0.9785', '0.9760', '0.9756', '0.9750', '0.9737', '0.9729', '0.9724', '0.9694', '0.9689', '0.9677', '0.9658', '0.9609', '0.9551', '0.9532', '0.9519', '0.9410', '0.9289', '0.9213', '0.9183', '0.9150', '0.9150', '0.9135', '0.9054', '0.9048', '0.9018', '0.9004', '0.8537', '0.6980', '0.6891', '0.6571', '0.6310', '0.5397', '0.4462', '0.3070', '0.2577', '0.2026', '0.1242', '0.1102', '0.0930', '0.0896', '0.0876', '0.0386', '0.0352']
# sizes ['3.08', '7.37', '9.62', '11.87', '16.36', '18.42', '20.27', '23.34', '25.16', '26.27', '26.92', '28.41', '30.23', '33.07', '35.85', '36.18', '37.66', '38.58', '41.54', '42.31', '43.69', '45.96', '47.74', '49.21', '49.63', '51.70', '52.65', '53.81', '56.03', '58.42', '59.49', '62.15', '64.06', '65.27', '67.23', '67.55', '68.69', '69.22', '71.23', '72.23', '73.97', '75.67', '76.60', '76.95', '77.40', '78.32', '78.91', '80.06', '81.58', '81.95', '82.44', '83.99', '85.13', '85.26', '85.76', '86.48', '87.29', '87.85', '88.90', '89.44', '89.80', '90.67', '91.43', '92.48', '93.28', '95.25', '96.16', '97.21', '98.72', '100.00']

low = 1  # (11.92)
low = 3  # (11.91)
low = 4  # (11.87)

low = 2  # (27.84)
low = 6  # (26.92)
low = 11  # (26.92)

low = 3  # (33.66)
low = 8  # (34.32)
low = 14  # (33.07)


# ####################
numberOfClusters = 15
numberOfClusters = 35
numberOfClusters = 70
rng = RandomState(randomSeed)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=rng).fit(aTrain)
sort_order = kutil.get_cluster_sen_ratios_sort_order(aTrain, linesTrain, kmeans)
y1 = kutil.get_cluster_sen_ratios(aTrain, linesTrain, kmeans, sort_order)
y1dev = kutil.get_cluster_sen_ratios(aDev, linesDev, kmeans, sort_order)
y2 = kutil.getScaledSizes(aTrain, kmeans, sort_order)
y2dev = kutil.getScaledSizes(aDev, kmeans, sort_order)

# numberOfClusters = 15
low = 1  # (11.92)
low = 2  # (27.84)
low = 3  # (33.66)

# numberOfClusters = 35
low = 3  # (11.91)  (trees_ijcai18_exp199_200K_10p.zip)
low = 6  # (26.92)  (trees_ijcai18_exp199_200K_25p.zip)
low = 8  # (34.32)

# numberOfClusters = 70
low = 4  # (11.87)
low = 11  # (26.92)
low = 14  # (33.07)

# ###############
clusterIds = sort_order  # clusterIds == sort_order, it's just syntaxtic sugar
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesTrainFull, aTrainFull, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesTrainFull, aTrainFull, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], linesDev, aDev, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:], linesDev, aDev, kmeans)
print("(numberOfClusters, low, acc, acc[c+1], fraction) = ({}, {}, {:.4f}%, {:.4f}%, {:.4f}%)".format(numberOfClusters, low, (1 - y1[low - 1]) * 100, (1 - y1[low]) * 100, y2[low - 1] * 100))
print("(devAcc, devFraction) = ({:.4f}%, {:.4f}%)".format((1 - y1dev[low - 1]) * 100, y2[low - 1] * 100))
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

# ###################################################
# ###################################################
# k=15  #############################################
# #############################################
#
# (numberOfClusters, low, acc, acc[c+1], fraction) = (15, 1, 99.4827%, 98.8064%, 11.9217%)
# (devAcc, devFraction) = (99.1468%, 11.9217%)
# 12583 91673
# 1758 13242
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9942 (0.9942), f1=0.0000 (0 0 12510 73)
# Accuracy (train C2): 0.9508 (0.8517), f1=0.8199 (10269 1181 76894 3329)
# Accuracy (dev C1): 0.9915 (0.9915), f1=0.0000 (0 0 1743 15)
# Accuracy (dev C2): 0.9323 (0.8510), f1=0.7536 (1370 293 10976 603)
# formula:  (+ (* (/ 1758.0 15000) 0.9914675767918089) (* (/ 13242.0 15000) 0.9323365050596587))
# k=15, c=1 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k15_c1.zip C1.txt C2.txt 2C1.txt 2C2.txt

# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (15, 2, 98.8064%, 98.2541%, 27.8400%)
# (devAcc, devFraction) = (98.1140%, 27.8400%)
# 29217 75039
# 4091 10909
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9900 (0.9901), f1=0.0000 (0 3 28924 290)
# Accuracy (train C2): 0.9428 (0.8217), f1=0.8272 (10269 1178 60480 3112)
# Accuracy (dev C1): 0.9856 (0.9856), f1=0.0000 (0 0 4032 59)
# Accuracy (dev C2): 0.9219 (0.8232), f1=0.7628 (1370 293 8687 559)
# formula:  (+ (* (/ 4091.0 15000) 0.9855780982644831) (* (/ 10909.0 15000) 0.921899349161243))
# k=15, c=2 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k15_c2.zip C1.txt C2.txt 2C1.txt 2C2.txt

# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (15, 3, 98.2541%, 98.2418%, 33.6633%)
# (devAcc, devFraction) = (97.1366%, 33.6633%)
# 35306 68950
# 4999 10001
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9886 (0.9886), f1=0.0000 (0 3 34902 401)
# Accuracy (train C2): 0.9394 (0.8075), f1=0.8309 (10269 1178 54502 3001)
# Accuracy (dev C1): 0.9830 (0.9830), f1=0.0000 (0 0 4914 85)
# Accuracy (dev C2): 0.9174 (0.8097), f1=0.7684 (1370 293 7805 533)
# formula:  (+ (* (/ 4999.0 15000) 0.9829965993198639) (* (/ 10001.0 15000) 0.9174082591740826))
# k=15, c=3 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k15_c3.zip C1.txt C2.txt 2C1.txt 2C2.txt

# ###################################################
# ###################################################
# k=35  #############################################
# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (35, 3, 99.4461%, 99.4165%, 11.9133%)
# (devAcc, devFraction) = (98.3165%, 11.9133%)
# 12531 91725
# 1696 13304
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9951 (0.9951), f1=0.0000 (0 0 12470 61)
# Accuracy (train C2): 0.9507 (0.8516), f1=0.8196 (10269 1181 76934 3341)
# Accuracy (dev C1): 0.9906 (0.9906), f1=0.0000 (0 0 1680 16)
# Accuracy (dev C2): 0.9327 (0.8518), f1=0.7538 (1370 293 11039 602)
# formula:  (+ (* (/ 1696.0 15000) 0.9905660377358491) (* (/ 13304.0 15000) 0.9327269993986771))
# k=35 c=3 trees_ijcai18_exp199_200K_10p.zip

# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (35, 6, 98.6283%, 98.5864%, 26.9167%)
# (devAcc, devFraction) = (98.1670%, 26.9167%)
# 28189 76067
# 3890 11110
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9933 (0.9933), f1=0.0000 (0 0 28001 188)
# Accuracy (train C2): 0.9422 (0.8227), f1=0.8237 (10269 1181 61403 3214)
# Accuracy (dev C1): 0.9884 (0.9884), f1=0.0000 (0 0 3845 45)
# Accuracy (dev C2): 0.9221 (0.8251), f1=0.7598 (1370 293 8874 573)
# formula:  (+ (* (/ 3890.0 15000) 0.9884318766066839) (* (/ 11110.0 15000) 0.922052205220522))
# k=35 c=6 trees_ijcai18_exp199_200K_25p.zip

# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (35, 8, 98.5765%, 98.3837%, 34.3217%)
# (devAcc, devFraction) = (98.5386%, 34.3217%)
# 35959 68297
# 5008 9992
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9914 (0.9914), f1=0.0000 (0 2 35649 308)
# Accuracy (train C2): 0.9374 (0.8043), f1=0.8278 (10269 1179 53755 3094)
# Accuracy (dev C1): 0.9868 (0.9868), f1=0.0000 (0 0 4942 66)
# Accuracy (dev C2): 0.9154 (0.8076), f1=0.7643 (1370 293 7777 552)
# formula:  (+ (* (/ 5008.0 15000) 0.9868210862619808) (* (/ 9992.0 15000) 0.9154323458767014))
# k=35 c=8 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k35_c8.zip C1.txt C2.txt 2C1.txt 2C2.txt

# ###################################################
# ###################################################
# k=70  #############################################
# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (70, 4, 99.5546%, 99.4801%, 11.8667%)
# (devAcc, devFraction) = (99.3789%, 11.8667%)
# 12538 91718
# 1712 13288
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9959 (0.9959), f1=0.0000 (0 0 12487 51)
# Accuracy (train C2): 0.9506 (0.8515), f1=0.8192 (10269 1181 76917 3351)
# Accuracy (dev C1): 0.9953 (0.9953), f1=0.0000 (0 0 1704 8)
# Accuracy (dev C2): 0.9320 (0.8510), f1=0.7521 (1370 293 11015 610)
# formula:  (+ (* (/ 1712.0 15000) 0.9953271028037384) (* (/ 13288.0 15000) 0.9320439494280554))

# k=70 c=4 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k70_c4.zip C1.txt C2.txt 2C1.txt 2C2.txt

# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (70, 11, 99.2386%, 99.2161%, 26.9233%)
# (devAcc, devFraction) = (96.2617%, 26.9233%)
# 28299 75957
# 3973 11027
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9946 (0.9946), f1=0.0130 (1 0 28146 152)
# Accuracy (train C2): 0.9417 (0.8220), f1=0.8225 (10268 1181 61258 3250)
# Accuracy (dev C1): 0.9930 (0.9930), f1=0.0000 (0 0 3945 28)
# Accuracy (dev C2): 0.9199 (0.8223), f1=0.7563 (1370 293 8774 590)
# formula:  (+ (* (/ 3973.0 15000) 0.9929524288950415) (* (/ 11027.0 15000) 0.9199238233427043))

# k=70 c=11 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k70_c11.zip C1.txt C2.txt 2C1.txt 2C2.txt

# #############################################
#

# (numberOfClusters, low, acc, acc[c+1], fraction) = (70, 14, 99.1192%, 99.1013%, 33.0700%)
# (devAcc, devFraction) = (97.2010%, 33.0700%)
# 34701 69555
# 4847 10153
# Accuracy (train): 0.9560 (0.8689), f1=0.8176 (10269 1181 89404 3402)
# Accuracy (dev): 0.9393 (0.8675), f1=0.7505 (1370 293 12719 618)
# Accuracy (train C1): 0.9939 (0.9939), f1=0.0093 (1 0 34488 212)
# Accuracy (train C2): 0.9372 (0.8065), f1=0.8245 (10268 1181 54916 3190)
# Accuracy (dev C1): 0.9901 (0.9901), f1=0.0000 (0 0 4799 48)
# Accuracy (dev C2): 0.9150 (0.8089), f1=0.7605 (1370 293 7920 570)
# formula:  (+ (* (/ 4847.0 15000) 0.9900969671962039) (* (/ 10153.0 15000) 0.9150004924652811))

# k=70 c=14 zip -m ../taboo-jan/functionality/201/trees_ijcai18_exp199_200K_k70_c14.zip C1.txt C2.txt 2C1.txt 2C2.txt

