# -*- coding: utf-8 -*-
"""

Created on September 21, 2017

@author:  neerbek
"""

from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
import numpy  # type: ignore
import theano  # type: ignore
import theano.tensor as T  # type: ignore
from datetime import datetime
import pylab  # type: ignore

import rnn_model.rnn
import rnn_model.learn
import hist
import confusion_matrix
import similarity.load_trees as load_trees
import embedding
import kmeans_cluster_util as kutil

inputfile = "../../taboo-core/output_embeddings.txt"

lines = confusion_matrix.read_embeddings(inputfile, max_line_count=-1)
lines_orig = list(lines)
#
lines = list(lines_orig)  # for recovering original list
rng = RandomState(8725)
perm = rng.permutation(len(lines))
lines = [lines[i] for i in perm]
lines2 = lines[14937:]
lines = lines[:14937]
print(len(lines), len(lines2))

# ## jump to dist of sen ratios
# ## to use dev instead
# save previous lines2
lines.extend(lines2)
len(lines)

inputfile = "../../taboo-core/output_embeddings2.txt"
lines2 = confusion_matrix.read_embeddings(inputfile, max_line_count=-1)
# End load of dev set

a = confusion_matrix.get_embedding_matrix(lines, normalize=True)
confusion_matrix.verify_matrix_normalized(a)

a2 = confusion_matrix.get_embedding_matrix(lines2, normalize=True)
confusion_matrix.verify_matrix_normalized(a2)

# ## attempt with both data sets
lines3 = [l for l in lines]
lines3.extend(lines2)
rng = RandomState(57478)
perm = rng.permutation(len(lines3))
lines3 = [lines3[i] for i in perm]

a3 = confusion_matrix.get_embedding_matrix(lines3, normalize=True)

rng = RandomState(2)
kmeans = KMeans(n_clusters=140, random_state=rng).fit(a3)


# ## plotting #################################################
rng = RandomState(2)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
# kmeans = KMeans(n_clusters=5, random_state=rng).fit(aC3)
# sort_order = kutil.get_cluster_sen_ratios_sort_order(aC3, linesC3, kmeans)
# y1 = kutil.get_cluster_sen_ratios(aC3, linesC3, kmeans, sort_order)
# get_cluster_sizes(aC3, kmeans)
# kutil.is_sorted(y1)
# y2 = kutil.get_cluster_sen_ratios(a2C3, lines2C3, kmeans, sort_order)

kmeans = KMeans(n_clusters=3, random_state=rng).fit(a)
kmeans = KMeans(n_clusters=10, random_state=rng).fit(a)
kmeans = KMeans(n_clusters=35, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)
y1 = kutil.get_cluster_sen_ratios(a, lines, kmeans, sort_order)
kutil.is_sorted(y1)
y2 = kutil.get_cluster_sen_ratios(a2, lines2, kmeans, sort_order)

rng = RandomState(387)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
kmeans = KMeans(n_clusters=35, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)
y3 = kutil.get_cluster_sen_ratios(a, lines, kmeans, sort_order)
kutil.is_sorted(y3)
y4 = kutil.get_cluster_sen_ratios(a2, lines2, kmeans, sort_order)

rng = RandomState(910)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
kmeans = KMeans(n_clusters=35, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)
y5 = kutil.get_cluster_sen_ratios(a, lines, kmeans, sort_order)
kutil.is_sorted(y5)
y6 = kutil.get_cluster_sen_ratios(a2, lines2, kmeans, sort_order)

rng = RandomState(2094)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
kmeans = KMeans(n_clusters=35, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)
y7 = kutil.get_cluster_sen_ratios(a, lines, kmeans, sort_order)
kutil.is_sorted(y7)
y8 = kutil.get_cluster_sen_ratios(a2, lines2, kmeans, sort_order)

rng = RandomState(48974)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
kmeans = KMeans(n_clusters=35, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)
y9 = kutil.get_cluster_sen_ratios(a, lines, kmeans, sort_order)
kutil.is_sorted(y9)
y10 = kutil.get_cluster_sen_ratios(a2, lines2, kmeans, sort_order)

y11 = [(y1[i] + y3[i] + y5[i] + y7[i] + y9[i]) / 5 for i in range(len(y1))]
y12 = [(y2[i] + y4[i] + y6[i] + y8[i] + y10[i]) / 5 for i in range(len(y1))]


x = range(len(y1))
x = range(len(y3))
confusion_matrix.new_graph('Clusters', 'Sensitive Ratio')
pylab.plot(x, y1, 'g:', label='Train $1$')
pylab.plot(x, y2, '-g', label='Dev $1$')
pylab.plot(x, y3, 'm:', label='Train $2$')
pylab.plot(x, y4, '-m', label='Dev $2$')
pylab.plot(x, y5, 'y:', label='Train $3$')
pylab.plot(x, y6, '-y', label='Dev $3$')
pylab.plot(x, y7, 'c:', label='Train $4$')
pylab.plot(x, y8, '-c', label='Dev $4$')
pylab.plot(x, y9, 'r:', label='Train $5$')
pylab.plot(x, y10, '-r', label='Dev $5$')
pylab.plot(x, y11, 'k:', label='Average Train')
pylab.plot(x, y12, '-k', label='Average Dev')
pylab.legend()
pylab.show()
pylab.savefig('../../../ExperimentDescription/figures/kmeans_sen_ratios_3_1train*' + '.eps')

# ## next step, training models for groups
rng = RandomState(7486)
(lines, a, lines2, a2) = kutil.generate_new_split(lines, lines2, rng)
kmeans = KMeans(n_clusters=35, random_state=rng).fit(a)
sort_order = kutil.get_cluster_sen_ratios_sort_order(a, lines, kmeans)

# cut [:16] + [16:28] + [28:]
# cut [:14] + [14:29] + [29:]
# cut [14:27]
low = 1
high = 2
low = 4
high = 8
low = 14
high = 29
clusterIdRange = range(len(sort_order))
clusterIds = [clusterIdRange[i] for i in sort_order]
(linesC1, aC1) = kutil.get_sentences_from_clusters(clusterIds[:low], lines, a, kmeans)
(linesC2, aC2) = kutil.get_sentences_from_clusters(clusterIds[low:high], lines, a, kmeans)
(linesC3, aC3) = kutil.get_sentences_from_clusters(clusterIds[high:], lines, a, kmeans)
(lines2C1, a2C1) = kutil.get_sentences_from_clusters(clusterIds[:low], lines2, a2, kmeans)
(lines2C2, a2C2) = kutil.get_sentences_from_clusters(clusterIds[low:high], lines2, a2, kmeans)
(lines2C3, a2C3) = kutil.get_sentences_from_clusters(clusterIds[high:], lines2, a2, kmeans)
print(len(linesC1), len(linesC2), len(linesC3))
print(len(lines2C1), len(lines2C2), len(lines2C3))
# 6061 7012 1864
# 1214 1416 370

kutil.get_base_accuracy(lines, "train").report()
kutil.get_base_accuracy(lines2, "validation").report()
cm = kutil.get_base_accuracy(linesC1, "train C1")
cm.report()
kutil.get_base_accuracy(lines2C1, "validation C1").report()
kutil.get_base_accuracy(linesC2, "train C2").report()
kutil.get_base_accuracy(lines2C2, "validation C2").report()
kutil.get_base_accuracy(linesC3, "train C3").report()
kutil.get_base_accuracy(lines2C3, "validation C3").report()

kutil.report_class_accuracy(0, low, linesC1)
kutil.report_class_accuracy(low, high, linesC2)
kutil.report_class_accuracy(high, "\infty", linesC3)

# choose data set
x_val = numpy.copy(a)  # we add column to  x_val
linesIn = lines
validation_x = numpy.copy(a2)  # we add bias
lines2In = lines2

x_val = numpy.copy(aC1)  # we add column to  x_val
linesIn = linesC1
validation_x = numpy.copy(a2C1)  # we add bias
lines2In = lines2C1

x_val = numpy.copy(aC2)  # we add column to  x_val
linesIn = linesC2
validation_x = numpy.copy(a2C2)  # we add bias
lines2In = lines2C2

x_val = numpy.copy(aC3)  # we add column to  x_val
linesIn = linesC3
validation_x = numpy.copy(a2C3)  # we add bias
lines2In = lines2C3

n = x_val.shape[0]
tmp = numpy.ones(shape=(n, x_val.shape[1] + 1))
tmp[:, 1:] = x_val
x_val = tmp

tmp = numpy.ones(shape=(validation_x.shape[0], validation_x.shape[1] + 1))
tmp[:, 1:] = validation_x
validation_x = tmp

y_val = numpy.zeros(shape=(n, 2))
for i in range(len(linesIn)):
    if linesIn[i].ground_truth == 4:
        y_val[i, 1] = 1
    else:
        y_val[i, 0] = 1

y_val_svm = numpy.zeros(shape=(n, ))
for i in range(len(linesIn)):
    y_val_svm[i] = linesIn[i].ground_truth

x_val = x_val.astype(dtype=theano.config.floatX)
y_val = y_val.astype(dtype=theano.config.floatX)
y_val_svm = y_val_svm.astype(dtype=theano.config.floatX)

validation_y = numpy.zeros(shape=(validation_x.shape[0], 2))
for i in range(len(lines2In)):
    if lines2In[i].ground_truth == 4:
        validation_y[i, 1] = 1
    else:
        validation_y[i, 0] = 1

validation_y_svm = numpy.zeros(shape=(validation_x.shape[0], ))
for i in range(len(lines2In)):
    validation_y_svm[i] = lines2In[i].ground_truth

validation_x = validation_x.astype(dtype=theano.config.floatX)
validation_y = validation_y.astype(dtype=theano.config.floatX)
validation_y_svm = validation_y_svm.astype(dtype=theano.config.floatX)

# ## SVM
# from sklearn import svm
# from sklearn import preprocessing
# # tried with different kernels, still bad. Internet says linear and rbf are good first attempts
# svc = svm.SVC(kernel='rbf')
# svc = svm.SVC(kernel='linear')
# svc.kernel

# # tried scaling, but saw no/little difference in my predictions (I do already normalized)
# scaler = preprocessing.StandardScaler()
# scaler = scaler.fit(x_val)

# x_val = scaler.transform(x_val)
# validation_x = scaler.transform(validation_x)
# svc.fit(x_val, y_val_svm)
# # res = svc.predict(validation_x)
# # res.shape
# # res[0]

# # get_cmcounter works on any model which implements predict(...)
# cm = get_cmcounter(svc, x_val, linesIn, "on train meta class set")
# cm.report()
# cm = get_cmcounter(svc, validation_x, lines2In, "on validation meta class subset")
# cm.report()


X = T.matrix('x', dtype=theano.config.floatX)
Y = T.matrix('y', dtype=theano.config.floatX)
Z = T.matrix('z', dtype=theano.config.floatX)

rng = RandomState(28)


rep_size = x_val.shape[1]
reg = rnn_model.rnn.Regression(RandomState(28), X, rep_size, 2)  # single layer, 2 outputs
params = reg.params
c = reg.cost(Y)
grads = [T.grad(cost=c, wrt=param) for param in params]
lr = 4.1  # if you change this, you have to recalculate updates (below)
updates = rnn_model.learn.gd(params=params, grads=grads, lr=lr)
retain_probability = 0.9
n_epochs = 2500
batch_size = 100
n_train_batches = int(numpy.around(n / batch_size + 0.5))

vali = reg.errors(Y)

update_keys = [k for k in updates.keys()]
outputs = [vali, c] + [updates[k] for k in update_keys]

train = theano.function(
    inputs=[X, Y],
    outputs=outputs)

for epoch in range(n_epochs):
    perm = rng.permutation(n)
    x_val = [x_val[i, :].reshape(1, rep_size) for i in perm]
    y_val = [y_val[i, :].reshape(1, 2) for i in perm]
    x_val = numpy.concatenate(x_val)
    y_val = numpy.concatenate(y_val)
    train_cost = 0
    train_acc = 0
    train_count = 0
    for minibatch_index in range(n_train_batches):
        x_in = x_val[minibatch_index * batch_size: (minibatch_index + 1) * batch_size, :]
        y_in = y_val[minibatch_index * batch_size: (minibatch_index + 1) * batch_size, :]
        current_size = x_in.shape[0]
        # z_in = rng.binomial(n=1, size=(current_size, rep_size), p=retain_probability)
        # z_in = z_in.astype(dtype=theano.config.floatX)
        values = train(x_in, y_in)
        train_acc += (1 - values[0]) * current_size
        train_cost += values[1] * current_size
        train_count += current_size
        for index, param in enumerate(update_keys):
            param.set_value(values[index + 2])
    if epoch % 10 == 0:
        print("{} Epoch {}. On train set : Node count {}, avg cost {:.6f}, avg acc {:.4f}%".format(
            datetime.now().strftime('%d%m%y %H:%M'), epoch, train_count,
            train_cost / train_count, train_acc / train_count * 100.))
    if epoch % 40 == 0:
        values = train(validation_x, validation_y)  # hack: we use train, really we are only evaluating
        current_size = validation_x.shape[0]
        train_acc = (1 - values[0]) * current_size
        train_cost = values[1] * current_size
        train_count = current_size
        print("{} Epoch {}. On Validation set : Node count {}, cost {:.6f}, acc {:.4f}%".format(
            datetime.now().strftime('%d%m%y %H:%M'), epoch, train_count,
            train_cost / train_count, train_acc / train_count * 100.))


# ## KMEANS
# remember to normalize $a$ first

# rng = RandomState(8726)
# kmeans = KMeans(n_clusters=140, random_state=rng).fit(a)
# n_clusters=170, many small clusters, suggests overfitting

# ## find all sentences (index) which belong to 'label' cluster
def print_cluster(label=2, lines=lines, kmeans=kmeans, max_count=100, rng=rng):
    idx = []
    perm = rng.permutation(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        index = perm[i]
        if kmeans.labels_[index] == label:
            idx.append(index)
    print(len(idx))
    # output max_count first elements in 'label' cluster
    for i in range(min(max_count, len(idx))):
        line = lines[idx[i]]
        sentence = load_trees.output_sentence(line.tree)
        print(i, idx[i], sentence)

# ## check a fraction of lines to see if they are assigned to the correct cluster
def check_clustering(kmeans=kmeans, lines=lines, count=250):
    perm = rng.permutation(len(lines))
    for i in range(count):
        index = perm[i]
        line = lines[index]
        emb = numpy.array(line.emb)
        emb = emb / numpy.max(emb)
        emb = emb / numpy.sqrt(numpy.sum(emb * emb))
        cluster_dist = 10000000
        cluster_index = 0
        for j in range(len(kmeans.cluster_centers_)):
            centroid = kmeans.cluster_centers_[j]
            # sim = embedding.cos_distance_numpy_vector(centroid, emb)
            dist = embedding.euclid_distance_numpy_vector(centroid, emb)
            if dist < cluster_dist:
                cluster_dist = dist
                cluster_index = j
        if cluster_index != kmeans.labels_[index]:
            print("error cluster_index seems wrong for sentence number: {}, calculated cluster: {}, kmeans cluster: {}".format(index, cluster_index, kmeans.labels_[index]))
    print("done")


# ## dist of label counts
res = kmeans.predict(a)
print(res.shape)
label_count_sen = [0 for i in range(len(kmeans.cluster_centers_))]
label_count_both = [0 for i in range(len(kmeans.cluster_centers_))]
for i in range(len(res)):
    l = res[i]
    line = lines[i]
    if line.ground_truth == 4:
        label_count_sen[l] += 1
    label_count_both[l] += 1
print(len(label_count_both))

# label_count_sen = [0 for i in range(len(kmeans.cluster_centers_))]
# label_count_both = [0 for i in range(len(kmeans.cluster_centers_))]
# for i in range(len(kmeans.labels_)):
#     l = kmeans.labels_[i]
#     line = lines[i]
#     if line.ground_truth == 4:
#         label_count_sen[l] += 1
#     label_count_both[l] += 1
# print(len(label_count_both))
# print(len(label_count_sen))

bincounts = hist.put_in_buckets(label_count_both, buckets=10)
print(bincounts)
bincounts = hist.put_in_buckets(label_count_sen, buckets=10)
print(bincounts)
print(label_count_both)
print(label_count_sen)
min(label_count_sen)
max(label_count_sen)
ratio = [kutil.stable_div(label_count_sen[i], label_count_both[i]) for i in range(len(label_count_both))]
print(["{:.3f}".format(r) for r in ratio])
bincounts = hist.put_in_buckets(ratio, buckets=20)
print(bincounts)

# ## check distribution of sen ratios in clusters
c = 0
fraction = 0.04
indexes = []
for i in range(len(label_count_sen)):
    r = kutil.stable_div(label_count_sen[i], label_count_both[i])
    if r > 1 - fraction:
        c += 1
        indexes.append(i)
    elif r < fraction:
        c += 1
        indexes.append(i)
print("Fraction {}. ratio is {}/{}. Approx {}".format(1 - fraction, c, len(kmeans.cluster_centers_), c / len(kmeans.cluster_centers_)))
len(indexes)
# Fraction 0.9. ratio is 99/170. Approx 0.58
# Fraction 0.95. ratio is 77/170. Approx 0.45
# Fraction 0.96. ratio is 74/170. Approx 0.43
# Fraction 0.97. ratio is 69/170. Approx 0.40
# Fraction 0.98. ratio is 62/170. Approx 0.36
# Fraction 0.99. ratio is 42/170. Approx 0.24

# import pylab
# x = range(len(label_count_sen))
# y1 = label_count_sen
#
# confusion_matrix.new_graph()
# pylab.plot(x, y1, 'g:')
# pylab.xlabel('Clusters')
# pylab.ylabel('Counts')
# pylab.legend()
# pylab.show()

#
# ## dev set
res = kmeans.predict(a2)
res.shape
label_count_res = [0 for i in range(len(kmeans.cluster_centers_))]
label_count_res_sen = [0 for i in range(len(kmeans.cluster_centers_))]
for i in range(len(res)):
    l = res[i]
    line = lines2[i]
    if line.ground_truth == 4:
        label_count_res_sen[l] += 1
    label_count_res[l] += 1
print(len(label_count_res))

c = 0
c2 = 0
atol = 0.05
for i in range(len(kmeans.cluster_centers_)):
    r = kutil.stable_div(label_count_sen[i], label_count_both[i])
    r2 = kutil.stable_div(label_count_res_sen[i], label_count_res[i])
    if not numpy.isclose(r, r2, atol=atol):
        print("Fraction differs. i={}, r={}, r2={}, count_res={}".format(i, r, r2, label_count_res[i]))
        c += 1
        if label_count_res[i] > 10:
            c2 += 1
print("Atol={} Number of diffs: {}. Number where |size|>10: {}".format(atol, c, c2))
# train split
# Atol=0.05 Number of diffs: 54  # a2 is split from training set
#
# train split. where clustering was done on concatenated both sets
# Atol=0.2 Number of diffs: 7
# Atol=0.05 Number of diffs: 55. Number where |size|>10: 26
#
# dev set
# Atol=0.01 Number of diffs: 147
# Atol=0.05 Number of diffs: 73
# Atol=0.1 Number of diffs: 51
# Atol=0.2 Number of diffs: 20
#
# dev set where clustering was done on dev+train
# Atol=0.05 Number of diffs: 71
# Atol=0.1 Number of diffs: 41
# Atol=0.2 Number of diffs: 11

# ## check if fractions on train and dev are the same. Here we only
# ## look at the clusters with lowest and highest fraction (the
# ## indexes list generated above)


ratio = [kutil.stable_div(label_count_res_sen[i], label_count_res[i]) for i in range(len(label_count_res))]
bincounts = hist.put_in_buckets(ratio, buckets=10)
print(bincounts)


c = 0
atol = 0.01
for i in indexes:
    r = kutil.stable_div(label_count_sen[i], label_count_both[i])
    r2 = kutil.stable_div(label_count_res_sen[i], label_count_res[i])
    if not numpy.isclose(r, r2, atol=atol):
        if r < 0.5 and r2 < r:
            continue
        if r > 0.5 and r2 > r:
            continue
        print("Fraction differs. i={}, r={}, r2={}. len(res[i])={}".format(i, r, r2, label_count_res[i]))
        c += 1
print("Atol={} Number of diffs: {}/{}".format(atol, c, len(indexes)))

c = 0
count1 = len(lines)
count2 = len(lines2)
for i in range(len(kmeans.cluster_centers_)):
    r = label_count_both[i] / count1
    r2 = label_count_res[i] / count2
    if not numpy.isclose(r, r2, atol=0.005):
        print("Fraction of members differs. i={}, r={}, r2={}".format(i, r, r2))
        c += 1
print("Number of diffs: ", c)
# for 0.01 atol: Number of diffs: 0
# for 0.005 atol: Number of diffs: 2

ratio = [kutil.stable_div(label_count_res_sen[i], label_count_res[i]) for i in range(len(label_count_res))]
ratio = [kutil.stable_div(label_count_sen[i], label_count_both[i]) for i in range(len(label_count_res))]
bincounts = hist.put_in_buckets(ratio, buckets=24)
print(bincounts)

def get_accuracies(kmeans=kmeans, embeddings=a, lines=lines):
    pred = kmeans.predict(embeddings)
    correct = [0 for k in kmeans.cluster_centers_]
    zeros = [0 for k in kmeans.cluster_centers_]
    count = [0 for k in kmeans.cluster_centers_]
    for i in range(len(lines)):
        idx = pred[i]
        line = lines[i]
        if line.is_correct:
            correct[idx] += 1
        if line.ground_truth == 0:
            zeros[idx] += 1
        count[idx] += 1
    return (correct, count, zeros)


(correct, count, zeros) = get_accuracies(kmeans, a, lines)
label = 2
print("accuracy for label={}: {:.3f} ({:.3f})".format(label, correct[label] / count[label], zeros[label] / count[label]))

(correct, count, zeros) = get_accuracies(kmeans, a, lines)
ratio = [kutil.stable_div(correct[i], count[i]) for i in range(len(count))]
min(ratio)
bincounts = hist.put_in_buckets(ratio, buckets=5)
print(bincounts)

(correct, count, zeros) = get_accuracies(kmeans, a2, lines2)
ratio = [kutil.stable_div(correct[i], count[i]) for i in range(len(count))]
max(ratio)
bincounts = hist.put_in_buckets(ratio, buckets=10)
print(bincounts)
