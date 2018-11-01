#  -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:17

@author: neerbek
"""

import sys
# import threading
# import multiprocessing
# import theano
# import theano.tensor as T

import numpy  # type: ignore
from numpy.random import RandomState  # type: ignore

# os.chdir("../../taboo-core")
import ai_util
import embedding
import confusion_matrix
import my_thread
import similarity.load_trees as load_trees

# import importlib
# importlib.reload(confusion_matrix)
# importlib.reload(embedding)
# importlib.reload(my_thread)

max_line_count = -1
inputfile = None
totaltimer = ai_util.Timer("Total time: ")
evaltimer = ai_util.Timer("Eval time: ")
totaltimer.begin()


def syntax():
    print("""syntax: read_embeddings.py [-inputfile <file>][-max_line_count <int>]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
# here you can insert manual arglist if needed
# arglist = "read_embeddings -inputfile output_embeddings.txt".split(" ")
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]

    next_i = i + 2
    if setting == '-inputfile':
        inputfile = arg
    elif setting == '-max_line_count':
        max_line_count = int(arg)
    else:
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i

if inputfile == None:
    raise Exception("Need an inputfile")

def finding_cm(begin, end, d1, indices, lines, closest_count, queue):
    end = min(end, len(lines))
    reportStep = 250
    expectedSteps = int((end - begin) / reportStep) + 1
    currentStep = 0
    valid_cm = []
    for i in range(begin, end):
        cm = confusion_matrix.finding_cm(d1, indices, i, lines, closest_count=closest_count)
        valid_cm.append(cm)
        if i % reportStep == 0:
            currentStep += 1
            print("{}/{} ({} :: {} :: {})".format(currentStep, expectedSteps, begin, i, end))
    queue.put(valid_cm)
    print("done thread")


# Evaluate
evaltimer.begin()

lines = confusion_matrix.read_embeddings(inputfile, max_line_count)
rng = RandomState(34837)
rng = RandomState(8726)
perm = rng.permutation(len(lines))
lines = [lines[i] for i in perm]
lines2 = lines[14937:]
lines = lines[:14937]
print(len(lines), len(lines2))
a = confusion_matrix.get_embedding_matrix(lines, normalize=True)
confusion_matrix.verify_matrix_normalized(a)

# ## distance matrix
d1 = embedding.cos_distance_numpy_matrix(a)
if d1.shape != (len(lines), len(lines)):
    raise Exception("unexpected size for d1")
else:
    print("ok")
indices = embedding.get_sort_indices_asc(d1)  # slow
indices = embedding.get_sort_indices_desc(d1)

lines2 = confusion_matrix.read_embeddings(inputfile="output_embeddings2.txt")
rng = RandomState(8726)
perm = rng.permutation(len(lines2))
lines2 = [lines2[i] for i in perm]
a2 = confusion_matrix.get_embedding_matrix(lines2, normalize=True)
confusion_matrix.verify_matrix_normalized(a2)

d2 = embedding.cos_distance_numpy_two_matrices(a, a2)
if d2.shape != (len(lines), len(lines2)):
    raise Exception("distance2 unexpected shape")
# indices2 = embedding.get_sort_indices_asc(d2)
indices2_dev2train = embedding.get_sort_indices_asc(d2)
indices2_train2dev = embedding.get_sort_indices_asc(d2.T)
indices2_dev2train = embedding.get_sort_indices_desc(d2)
indices2_train2dev = embedding.get_sort_indices_desc(d2.T)

# ## Find matching nearest neightbors (takes time) and make ConfusionMatrix
closest_count = 800
valid_cm = my_thread.run(4, lines, target=finding_cm, args=(d1, indices, lines, closest_count), combiner=my_thread.extender)

# default impl: 900s
# 3 threads: 333s, (Queue) 321s
# 4 threads: (Queue) 237s, (my_thread) 157s, 130s, 139s (battery), 167s 
# 5 threads: (Queue) 241s, 235s
# 6 threads: 241s, 231 (Queue), 232, 253s, (my_thread) 251s, 250s
# 7 threads: 230s

len(valid_cm)

# ## Generate a ConfusionMatrixSimple (somewhat fast)
valid_cms = []
for i in range(len(lines)):
    cm = valid_cm[i]
    line = lines[i]
    score = line.sen_score
    ground_truth = line.ground_truth  # 0 if non-sensitive, 4 ow.
    cms = confusion_matrix.ConfusionMatrixSimple(index=i, score=score, ground_truth=ground_truth)
    cms.load_values(cm)
    valid_cms.append(cms)
    if i % 2000 == 0:
        print("{}/{}".format(i, len(lines)))
print(len(valid_cms))

# ## Find Thresholds and count them ####################
best = [0, 0]
start_index = 0
for index in range(start_index, len(valid_cm)):
    cm = valid_cm[index]
    (x, pos, neg) = confusion_matrix.find_threshold(cm)
    count = 0
    if cm.t_pos != None:
        count += pos[cm.t_pos][1]
    if cm.t_neg != None:
        count += neg[cm.t_neg][1]
    if count > best[0]:
        best[0] = count
        best[1] = index
    if index % 1000 == 0:
        print("{}/{}".format(index, len(valid_cm)))
print("Done threshold finding. Best was index={} with count={} (start_index {})".format(best[1], best[0], start_index))
# Asc (no suffling) #####################################
# Done threshold finding. Best was index=466 with count=38
# Done t_pos=167 t_neg=1371
# 800, 200 accuracy: 0.9888 count 17937 total 17937. Std_Preds 16441. New_preds 1496
# Asc (suffling, seed=34837) ############################
# Done threshold finding. Best was index=3 with count=1
#                       2-Best was index=7 with count=1
# Done t_pos=154 t_neg=1358
# 800, 200 accuracy: 0.9887 count 17937 total 17937. Std_Preds 16455. New_preds 1482
# Asc (suffling, seed=8726) ############################
# Done threshold finding. Best was index=5021 with count=20 (start_index 0)
# Done t_pos=154 t_neg=1371
# 800, 200 accuracy: 0.9891 count 17937 total 17937. Std_Preds 16445. New_preds 1492
# Desc ##################################################
# Done threshold finding. Best was index=4443 with count=87
# Done t_pos=2908 t_neg=7387
# 800, 160 accuracy: 0.9083 count 17937 total 17937. Std_Preds 17800. New_preds 137
# Desc (suffling, seed=8726) ############################
# Done threshold finding. Best was index=10871 with count=90 (start_index 0)
# Done t_pos=2902 t_neg=7389
# 800, 160 accuracy: 0.9086 count 17937 total 17937. Std_Preds 17804. New_preds 133
# Desc (suffling, seed=8726, normalized embeddings) ######################
# Done threshold finding. Best was index=10871 with count=90 (start_index 0)
# Done t_pos=2902 t_neg=7375
# 800, 160 accuracy: 0.9086 count 17937 total 17937. Std_Preds 17804. New_preds 133 (i.e. same)


# ## plot counts (from above) and the approximated normal distributions ##########
index = 7
cm = valid_cm[index]
cms = valid_cms[index]
cm.sentence
cm.report()
cms.score, cms.ground_truth
show_normal = []
for t in [confusion_matrix.ConfusionMatrix.TYPE_TP, confusion_matrix.ConfusionMatrix.TYPE_FP, confusion_matrix.ConfusionMatrix.TYPE_TN, confusion_matrix.ConfusionMatrix.TYPE_FN]:
    show_normal.append(cm.get_avg_score(t))
    show_normal.append(cm.get_std_dev_score(t))
    show_normal.append(len(cm.get_score(t)))

# import importlib
# importlib.reload(confusion_matrix)
confusion_matrix.plot_cm(cm, index, save=False, show=True, show_normal=show_normal, scale_equally=True, show_additive=False, scaleFactor=0.0000015)
confusion_matrix.plot_cm(cm, index, save=False, show=True, show_normal=None)

# ## Count number of valid thresholds found ##################
t_pos = 0
t_neg = 0
for index in range(len(valid_cm)):
    cm = valid_cm[index]
    if cm.t_pos != None:
        t_pos += 1
    if cm.t_neg != None:
        t_neg += 1
print("Done t_pos={} t_neg={}".format(t_pos, t_neg))

# ## Plot accumelated counts (for thresholds) #################
index = 4443
index = 8000
cm = valid_cm[index]
(x, pos, neg) = confusion_matrix.find_threshold(cm)

print(cm.t_pos, cm.t_neg)

y1 = [p[0] if p[0] < 100 else 100 for p in pos]
y2 = [p[1] if p[1] < 100 else 100 for p in pos]
y3 = [p[0] if p[0] < 100 else 100 for p in neg]
y4 = [p[1] if p[1] < 100 else 100 for p in neg]
confusion_matrix.plot_graphs(x, tp=y1, fp=y2, tn=y3, fn=y4, name=None, show=True)
confusion_matrix.plot_graphs(x, y1, y2, y3, y4, "acc_count_s{}".format(index))

###

# ## A check on ground truth values
for i in range(len(valid_cms)):
    cms = valid_cms[i]
    if cms.ground_truth != 0 and cms.ground_truth != 4:
        print("unexpected gt {}".format(cms.ground_truth))
        break
print("done")

def predict2(cms):
    self = cms
    if self.score < 0.5:
        t = confusion_matrix.ConfusionMatrix.TYPE_TN
        if self.is_within(t):
            return 0
    else:
        t = confusion_matrix.ConfusionMatrix.TYPE_TP
        if self.is_within(t):
            return 4
    element = self.scores[confusion_matrix.ConfusionMatrix.TYPE_FP]
    if element.count < 5:
        return self.get_standard_prediction()
    min_score = element.score - 2.5 * element.std_dev
    if self.score < min_score:
        return 0
    element = self.scores[confusion_matrix.ConfusionMatrix.TYPE_FN]
    max_score = element.score + 2.5 * element.std_dev
    if self.score > max_score:
        return 4
    return self.get_standard_prediction()

def predict3(cm, cms):
    self = cms
    if self.score < 0.5:
        if cm.t_neg is None:
            return 0
        if self.score * 100 > cm.t_neg:
            return 4
        else:
            return 0
    else:
        if cm.t_pos is None:
            return 4
        if self.score * 100 < cm.t_pos:
            return 0
        else:
            return 4

def predict3_val(cm, line):
    if line.sen_score < 0.5:
        if cm.t_neg is None:
            return 0
        if line.sen_score * 100 > cm.t_neg:
            return 4
        else:
            return 0
    else:
        if cm.t_pos is None:
            return 4
        if line.sen_score * 100 < cm.t_pos:
            return 0
        else:
            return 4

def predict4(index, indices, valid_cm, valid_cms, closest_count=100):
    index_indices = indices[:, index]
    pred = 0
    cm = valid_cm[index]
    for i in range(closest_count):
        best = index_indices[i]
        cms = valid_cms[best]
        pred += predict3(cm, cms)
        # pred += cms.get_standard_prediction()
    pred = pred / closest_count
    cms = valid_cms[index]
    if pred < 2:
        return 0
    if pred > 2:
        return 4
    return cms.get_standard_prediction()


# Standard Prediction: 0.9061 17937 17937
# Len cutoff <=1:      0.9061 17937 17937
# Len cutoff <=2:      0.9146 16717 17937
# Len cutoff <=3:      0.9173 15185 17937
# Len cutoff <=4:      0.9182 14187 17937
# Len cutoff <=5:      0.9189 13352 17937
# Len cutoff <=6:      0.9222 12379 17937
# Len cutoff <=7:      0.9238 11686 17937
# Len cutoff <=8:      0.9238 11056 17937
# Len cutoff <=9:      0.9247 10623 17937
# Len cutoff <=10:     0.9236 10041 17937
# Len cutoff <=11:     0.9232 9653 17937
# Len cutoff <=12:     0.9227 9338 17937

# closest_count=800
# 200, 600 accuracy: 0.9867 count 17937 total 17937. Std_Preds 16442. New_preds 1495
# 200, 500 accuracy: 0.9873 count 17937 total 17937. Std_Preds 16441. New_preds 1496
# 200, 400 accuracy: 0.9880 count 17937 total 17937. Std_Preds 16440. New_preds 1497
# 200, 300 accuracy: 0.9885 count 17937 total 17937. Std_Preds 16440. New_preds 1497
# 200, 200 accuracy: 0.9888 count 17937 total 17937. Std_Preds 16441. New_preds 1496
# 200, 100 accuracy: 0.9895 count 17937 total 17937. Std_Preds 16437. New_preds 1500
# 200, 50 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# \mathbf 200, 25 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 200, 1 accuracy: 0.9109 count 17937 total 17937. Std_Preds 17851. New_preds 86
# 300, 10 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 300, 10 accuracy: 0.9666 count 17937 total 17937. Std_Preds 16851. New_preds 1086
# 300, 50 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 400, 30 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 400, 10 accuracy: 0.9666 count 17937 total 17937. Std_Preds 16851. New_preds 1086
# 400, 16 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 400, 20 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 800, 20 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 800, 10 accuracy: 0.9666 count 17937 total 17937. Std_Preds 16851. New_preds 1086
# 800, 18 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 800, 11 accuracy: 0.9730 count 17937 total 17937. Std_Preds 16737. New_preds 1200
# 800, 14 accuracy: 0.9897 count 17937 total 17937. Std_Preds 16437. New_preds 1500
# 800, 16 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 1600, 16 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 1600, 15 accuracy: 0.9897 count 17937 total 17937. Std_Preds 16437. New_preds 1500
# 1600, 14 accuracy: 0.9897 count 17937 total 17937. Std_Preds 16437. New_preds 1500
# 1600, 13 accuracy: 0.9842 count 17937 total 17937. Std_Preds 16536. New_preds 1401
# 6400, 16 accuracy: 0.9896 count 17937 total 17937. Std_Preds 16436. New_preds 1501
# 6400, 14 accuracy: 0.9897 count 17937 total 17937. Std_Preds 16437. New_preds 1500
# 6400, 13 accuracy: 0.9842 count 17937 total 17937. Std_Preds 16536. New_preds 1401
#
# Len <= 2: 1600, 14 accuracy: 0.9891 count 16717 total 17937. Std_Preds 15472. New_preds 1245
#
#
# With correct ordering in indicies
# 200, 200 accuracy: 0.9082 count 17937 total 17937. Std_Preds 17800. New_preds 137
# 800, 160 accuracy: 0.9086 count 17937 total 17937. Std_Preds 17804. New_preds 133
# (with normalized embeddings) 800, 160 accuracy: 0.9086 count 17937 total 17937. Std_Preds 17804. New_preds 133
# 1000, 20 accuracy: 0.9081 count 17937 total 17937. Std_Preds 17798. New_preds 139
# 1000, 400 accuracy: 0.9083 count 17937 total 17937. Std_Preds 17799. New_preds 138
# 2000, 200 accuracy: 0.9083 count 17937 total 17937. Std_Preds 17799. New_preds 138
#
search_range = 800
match_range = 160

def standard_prediction(score):
    if score < 0.5:
        return 0
    return 4

def predict5(index, indices, valid_cm, valid_cms):
    index_indices = indices[:, index]  # get closest match to index
    cms_index = valid_cms[index]
    for i in range(search_range):
        current = index_indices[i]
        cm = valid_cm[current]
        index_indices2 = indices[:, current]  # get closest match to current
        # currentDist = 1
        for j in range(match_range):
            # currentDist = d1[index, index_indices2[j]]
            if index == index_indices2[j]:  # current line is in the top#match_range# of this line
                # print("currentDist was", currentDist)
                return predict3(cm, cms_index)
                # return predict4(index, indices, valid_cm, valid_cms, closest_count=5)
    # print("standard prediction: currentDist was", currentDist)
    return cms_index.get_standard_prediction()

def predict5_2(index, indices, valid_cm, valid_cms):
    index_indices = indices[:, index]  # get closest match to index
    cms_index = valid_cms[index]
    std_pred = cms_index.get_standard_prediction()
    for i in range(search_range):
        current = index_indices[i]
        cm = valid_cm[current]
        index_indices2 = indices[:, current]  # get closest match to current
        # currentDist = 1
        for j in range(match_range):
            # currentDist = d1[index, index_indices2[j]]
            if index == index_indices2[j]:  # current line is in the top#match_range# of this line
                # print("currentDist was", currentDist)
                pred = predict3(cm, cms_index)
                if pred != std_pred:
                    return pred
                break  # j loop, continue i loop
    # print("standard prediction: currentDist was", currentDist)
    return std_pred

def predict5_val(index, indices, indices2, lines2, valid_cm, d1, d2):
    index_indices = indices2[:, index]  # get closest match to index
    line = lines2[index]
    for i in range(search_range):
        current = index_indices[i]  # get index of closest match
        cm = valid_cm[current]
        distLineCurrent = d2[current, index]
        current_indices = indices[:, current]
        distCurrentWorst = d1[current, current_indices[match_range]]
        # print("worst", distCurrentWorst, "current", distLineCurrent)
        # many are 0, why?
        if distCurrentWorst <= distLineCurrent:  # similarity (not dist), less is worse
            return predict3_val(cm, line)
    # standard pred
    return standard_prediction(line.sen_score)

def predict5_val_asc(index, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm):
    index_indices = indices2_dev2train[:, index]  # get closest matches (from train) to index (from dev)
    line = lines2[index]
    for i in range(search_range):
        current = index_indices[i]  # get index of closest match (this is a train index)
        cm = valid_cm[current]
        current_indices = indices2_train2dev[:, current]  # get closest matches (from dev) to current (from train)
        for j in range(match_range):
            if index == current_indices[j]:  # current line (from dev) is in the top#match_range# of current (from train)
                return predict3_val(cm, line)
    return standard_prediction(line.sen_score)

def predict5_val_asc_where(index, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm):
    index_indices = indices2_dev2train[:, index]  # get closest matches (from train) to index (from dev)
    line = lines2[index]
    for i in range(search_range):
        current = index_indices[i]  # get index of closest match (this is a train index)
        cm = valid_cm[current]
        current_indices = indices2_train2dev[:, current]  # get closest matches (from dev) to current (from train)
        j = numpy.where(current_indices == index)[0]
        if j < 0.8 * match_range:
                return predict3_val(cm, line)
    return standard_prediction(line.sen_score)

def predict5_val_asc_nomatch(index, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm):
    index_indices = indices2_dev2train[:, index]  # get closest matches (from train) to index (from dev)
    line = lines2[index]
    for i in range(search_range):
        current = index_indices[i]  # get index of closest match (this is a train index)
        cm = valid_cm[current]
        return predict3_val(cm, line)
    return standard_prediction(line.sen_score)

def predict5d(index, indices, valid_cm, valid_cms):
    cms_index = valid_cms[index]
    current = 4756
    cm = valid_cm[current]
    index_indices2 = indices[:, current]  # get closest match to current
    for j in range(match_range):
        if index == index_indices2[j]:  # current line is in the top#match_range# of this line
            return predict3(cm, cms_index)
    return cms_index.get_standard_prediction()

def predict5c(index, indices, valid_cm, valid_cms):
    index_indices = indices[:, index]  # get closest match to index
    # TODO select random
    cms_index = valid_cms[index]
    best = [match_range, -1]
    for i in range(search_range):
        current = index_indices[i]
        cm = valid_cm[current]
        index_indices2 = indices[:, current]  # get closest match to current
        for j in range(match_range):
            if index == index_indices2[j]:  # current line is in the top#match_range# of this line
                if best[0] > j:
                    best[0] = j         # save lowest index (j) seen so far
                    best[1] = current   # save which search index (current) the best is for
    if best[1] != -1:
        cm = valid_cm[best[1]]
        return predict3(cm, cms_index)

    return cms_index.get_standard_prediction()


def predict5e(index, indices, valid_cm, valid_cms):
    index_indices = indices[:, index]  # get closest match to index
    # TODO select random
    cms_index = valid_cms[index]
    best = [match_range, -1]
    for i in range(search_range):
        current = index_indices[i]
        cm = valid_cm[current]
        index_indices2 = indices[:, current]  # get closest match to current
        for j in range(match_range):
            if index == index_indices2[j]:  # current line is in the top#match_range# of this line
                if best[0] > j:
                    best[0] = j         # save lowest index (j) seen so far
                    best[1] = current   # save which search index (current) the best is for
    if best[1] != -1:
        cm = valid_cm[best[1]]
        return predict3(cm, cms_index)

    return cms_index.get_standard_prediction()


# DEBUG
# index = 300
# index_indices = indices[index, :]
# t3 = embedding.apply_sort_indices(d1, indices)
# print(t3[index, :10])

def predict5b(index2, cms2, indices2, dist2, indices, dist):
    """ predict cms2 (from different dataset) and trainset (indices + dist)"""
    index_indices2 = indices2[:, index2]  # assume dists are ordered in column
    for i in range(search_range):
        current = index_indices2[i]
        index_indices = indices[:, current]
        cms2_dist = dist2[i, index2]  # assume dists are ordered in columns
        j = index_indices[match_range]  # jth (match_range) best match distance
        current_dist = dist[j, current]
        if current_dist > cms2_dist:  # cms2_dist to current is better (smaller) than jth match
            return predict3(cm, cms2)
    return cms2.get_standard_prediction()


def predict6(cms_index):
    return cms_index.get_standard_prediction()

def calc_accuracy_train(start_index, end_index, indices, valid_cm, valid_cms, queue):
    acc = 0
    std_pred_count = 0
    new_pred_count = 0
    count = 0
    for i in range(start_index, end_index):
        # if len(cm.sentence.split()) < 2:
        #     continue
        # pred = cms.predict()
        #  pred = predict3(cm, cms)
        # pred = predict4(i, indices, valid_cm, valid_cms, closest_count=30)
        pred = predict5(i, indices, valid_cm, valid_cms)
        cms = valid_cms[i]
        std_pred = standard_prediction(cms.score)
        if pred == std_pred:
            std_pred_count += 1
        else:
            new_pred_count += 1
        count += 1
        if pred == cms.ground_truth:
            acc += 1
        if i % 2000 == 0:
            print("{} :: {} :: {}".format(start_index, i, end_index))
    queue.put([acc, std_pred_count, new_pred_count, count])

def calc_accuracy_dev(start_index, end_index, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm, queue):
    acc = 0
    std_pred_count = 0
    new_pred_count = 0
    count = 0
    for i in range(start_index, end_index):
        line = lines2[i]
        # pred = predict5_val_asc_where(i, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm)
        pred = predict5_val_asc(i, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm)
        # pred = predict5_val_asc_nomatch(i, indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm)
        # pred = predict5_val(i, indices, indices2, lines2, valid_cm, d1, d2)
        std_pred = standard_prediction(line.sen_score)
        if pred == std_pred:
            std_pred_count += 1
        else:
            new_pred_count += 1
        count += 1
        if pred == line.ground_truth:
            acc += 1
        if i % 250 == 0:
            print("{} :: {} :: {}".format(start_index, i, end_index))
    queue.put([acc, std_pred_count, new_pred_count, count])

def calc_accuracy_combiner(res, current):
    if len(res) == 0:
        res = [0, 0, 0, 0]
    res[0] += current[0]
    res[1] += current[1]
    res[2] += current[2]
    res[3] += current[3]
    return res

def report_accuracy(res):
    (acc, std_pred_count, new_pred_count, count) = res
    if count == 0:
        print("accuracy: 0 {} {}".format(count, len(valid_cm)))
    else:
        print("{}, {} accuracy: {:.4f} count {} total {}. Std_Preds {}. New_preds {}".format(search_range, match_range, acc / count, count, len(valid_cm), std_pred_count, new_pred_count))


# train set
report_accuracy(my_thread.run(6, lines, target=calc_accuracy_train, args=(indices, valid_cm, valid_cms), combiner=calc_accuracy_combiner))

# dev set
report_accuracy(my_thread.run(6, lines2, target=calc_accuracy_dev, args=(indices, indices2_dev2train, indices2_train2dev, lines2, valid_cm), combiner=calc_accuracy_combiner))
# Standard accuracy: 0.8413 5981 5981
# Asc sorting
# MultiThread: Run (t=6): Number of updates: 1. Total time: 50.92. Average time: 50.9205 sec
# 800, 200     accuracy: 0.8417 count 5981 total 17937. Std_Preds 5973. New_preds 8
# 800, 200x2   accuracy: 0.8398 count 5981 total 17937. Std_Preds 5958. New_preds 23
# 800, 200x0.8 accuracy: 0.8418 count 5981 total 17937. Std_Preds 5974. New_preds 7
# 800, 160 accuracy:     0.8418 count 5981 total 17937. Std_Preds 5974. New_preds 7
# MultiThread: Run (t=6): Number of updates: 1. Total time: 253.35. Average time: 253.3495 sec
# 5000, 160 accuracy:    0.8418 count 5981 total 17937. Std_Preds 5974. New_preds 7
#
# Desc sorting
# MultiThread: Run (t=6): Number of updates: 1. Total time: 1.08. Average time: 1.0808 sec
# 800, 160 accuracy:     0.8366 count 5981 total 17937. Std_Preds 5907. New_preds 74
# 800, 100 accuracy: 0.8365 count 5981 total 17937. Std_Preds 5908. New_preds 73
# 800, 50 accuracy: 0.8363 count 5981 total 17937. Std_Preds 5907. New_preds 74
# 800, 25 accuracy: 0.8368 count 5981 total 17937. Std_Preds 5908. New_preds 73
# 800, 12 accuracy: 0.8368 count 5981 total 17937. Std_Preds 5910. New_preds 71
# 200, 12 accuracy: 0.8368 count 5981 total 17937. Std_Preds 5910. New_preds 71
# 5, 12 accuracy: 0.8370 count 5981 total 17937. Std_Preds 5911. New_preds 70
# no_match: 200, 12 accuracy: 0.8363 count 5981 total 17937. Std_Preds 5907. New_preds 74
# no_match: 2000, 12 accuracy: 0.8363 count 5981 total 17937. Std_Preds 5907. New_preds 74
# no_match: 5, 12 accuracy: 0.8363 count 5981 total 17937. Std_Preds 5907. New_preds 74

# importlib.reload(my_thread)
# print(my_thread.run(3, lines, my_thread.example_target, args=(), combiner=my_thread.adder))

i = 17921
cms = valid_cms[i]
cm = valid_cm[i]
pred = cms.predict()
print(i, cms.score, cms.ground_truth, pred, end='')
cm.report()
ind = indices[:, i]
for j in range(10):
    cm = valid_cm[ind[j]]  # jth best match
    cm.report()

# regular prediction ################################
acc = 0
count = 0
lineList = lines2
min_length = 0
for i in range(len(lineList)):
    line = lineList[i]
    sentence = load_trees.output_sentence(line.tree)
    score = line.sen_score
    if len(sentence.split()) < min_length:
        continue
    pred = standard_prediction(score)
    if pred == line.ground_truth:
        acc += 1
    count += 1
print("accuracy: {:.4f} {} {}".format(acc / count, count, len(lineList)))

# ## validation scores
#
# Standard Prediction: 0.8413 5981 5981
# Len cutoff <=1:      0.8437 5592 5981
# Len cutoff <=2:      0.8437 5061 5981
# Len cutoff <=3:      0.8443 4709 5981
# Len cutoff <=4:      0.8463 4418 5981
# Len cutoff <=6:      0.8539 3819 5981
# Len cutoff <=8:      0.8563 3480 5981
# Len cutoff <=10:     0.8524 3144 5981

# closest_count=800
# 200, 200 accuracy: 0.8071 count 5981 total 5981. Std_Preds 5380. New_preds 601
# 200, 14 accuracy: 0.8071 count 5981 total 5981. Std_Preds 5380. New_preds 601
#
# asc ordering
# 800, 14 accuracy: 0.8413 count 5981 total 5981. Std_Preds 5977. New_preds 4
# 800, 200 accuracy: 0.8397 count 5981 total 5981. Std_Preds 5963. New_preds 18
#
# asc ordering (shuffle: 8726)
# 800, 200 accuracy: 0.8417 count 5981 total 5981. Std_Preds 5973. New_preds 8


i = 131
line = lines[i]
score = line.sen_score
ground_truth = line.ground_truth  # 0 if non-sensitive
cm = confusion_matrix.finding_cm(d1, indices, i, lines, closest_count=300)
cm.report()
cms = confusion_matrix.ConfusionMatrixSimple(index=i, score=score, ground_truth=ground_truth)
cms.load_values(cm)
cms.score
cms.predict()
cms.ground_truth
cm.fn_score

evaltimer.end()

evaltimer.report()
totaltimer.end().report()


