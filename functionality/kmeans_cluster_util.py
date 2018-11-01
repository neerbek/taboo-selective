# -*- coding: utf-8 -*-
"""

Created on October 3, 2017

@author:  neerbek
"""
import numpy  # type: ignore
import confusion_matrix

# ################################################################################
#
# ## dist of sen ratios
#
def stable_div(a, b):
    if b == 0:
        return 0
    return a / b

def get_cluster_sizes(emb_matrix, kmeans):
    res = kmeans.predict(emb_matrix)
    label_count_both = [0 for i in range(len(kmeans.cluster_centers_))]
    for i in range(len(res)):
        l = res[i]
        label_count_both[l] += 1
    print("size is", numpy.sum(label_count_both))
    return label_count_both

def get_cluster_sen_ratios_impl(emb_matrix, lines, kmeans):
    """For each cluster calc ratio of #sensitive to cluster size
    returns: a list we the ratio for each cluster
    """
    res = kmeans.predict(emb_matrix)
    label_count_sen = [0 for i in range(len(kmeans.cluster_centers_))]
    label_count_both = [0 for i in range(len(kmeans.cluster_centers_))]
    for i in range(len(res)):
        l = res[i]
        line = lines[i]
        if line.ground_truth == 4:
            label_count_sen[l] += 1
        label_count_both[l] += 1
    print("size is", emb_matrix.shape[0])
    ratio = [stable_div(label_count_sen[i], label_count_both[i]) for i in range(len(label_count_both))]
    return ratio

def get_cluster_sen_ratios_sort_order(emb_matrix, lines, kmeans):
    ratio = get_cluster_sen_ratios_impl(emb_matrix, lines, kmeans)
    ratio = numpy.array(ratio)
    indexes = numpy.argsort(ratio)  # cluster_id to sort_index, asc
    return [i for i in indexes]

def get_cluster_sen_ratios(emb_matrix, lines, kmeans, sort_order):
    ratio = get_cluster_sen_ratios_impl(emb_matrix, lines, kmeans)
    return [ratio[i] for i in sort_order]  # return in sort_order order

def is_sorted(y1):
    prev = 0
    for y in y1:
        if y < prev:
            print("sorting is wrong for {}, {}".format(y, prev))
        prev = y

def generate_new_split(lines1, lines2, rng, cutoff=14937):
    """Takes lines1 and lines2 and combines, shuffles and split again. Useful for working with random splits of data"""
    lines = [l for l in lines1]  # lines1 may not be a list but rather iterable
    lines.extend(lines2)
    perm = rng.permutation(len(lines))
    lines = [lines[i] for i in perm]
    lines1 = lines[:cutoff]
    lines2 = lines[cutoff:]
    a1 = confusion_matrix.get_embedding_matrix(lines1, normalize=True)
    a2 = confusion_matrix.get_embedding_matrix(lines2, normalize=True)
    return (lines1, a1, lines2, a2)


""" Very simple ConfusionMatrix, only count members of different classes and report"""
class CMCounter:
    def __init__(self, name=None):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.name = name

    def report(self):
        acc = self.get_accuracy()
        f1 = 2 * self.tp / (2 * self.tp + self.fn + self.fp)
        prec = stable_div(self.tp, (self.tp + self.fp))
        reca = stable_div(self.tp, (self.tp + self.fn))
        f1_2 = stable_div(2 * prec * reca, (prec + reca))
        if not numpy.isclose(f1, f1_2, atol=0.0001):
            print("scores are not equal {} {}".format(f1, f1_2))

        pre_text = "Accuracy:"
        if self.name != None:
            pre_text = "Accuracy (" + self.name + "):"
        print(pre_text + " {:.4f} ({:.4f}), f1={:.4f} ({} {} {} {})".format(acc, (self.tn + self.fp) / self.get_sum(), f1, self.tp, self.fp, self.tn, self.fn))

    def get_sum(self):
        return self.tp + self.fp + self.tn + self.fn

    def get_accuracy(self):
        return (self.tp + self.tn) / self.get_sum()

def get_cmcounter(model, x, lines, name=None):
    res = model.predict(x)
    cm = CMCounter(name)
    for i in range(len(res)):
        ground_truth = lines[i].ground_truth
        if res[i] == 4:
            if ground_truth == 4:
                cm.tp += 1
            else:
                cm.fp += 1
        else:
            if ground_truth == 4:
                cm.fn += 1
            else:
                cm.tn += 1
    if len(res) != len(lines) or len(res) != cm.get_sum():
        print("something wrong in getting CMCounter")
    return cm

def get_base_accuracy(lines, name=None):
    cm = CMCounter(name)
    for i in range(len(lines)):
        line = lines[i]
        if line.is_correct:
            if line.ground_truth == 0:
                cm.tn += 1
            else:
                cm.tp += 1
        else:
            if line.ground_truth == 0:
                cm.fp += 1
            else:
                cm.fn += 1
    return cm

def report_class_accuracy(low, high, lines):
    cm = get_base_accuracy(lines)
    name = "$[{};{}[$".format(low, high)
    size = "${}$".format(len(lines))
    non_sen = "${:.4f}$".format((cm.tn + cm.fp) / cm.get_sum())
    acc = "${:.4f}$".format(cm.get_accuracy())
    confusion = "${}$ & ${}$ & ${}$ & ${}$ \\\\".format(cm.tp, cm.fp, cm.tn, cm.fn)
    print(name + " & " + size + " & " + non_sen + " & " + acc + " & " + confusion)

def get_sentences_from_clusters(clusterIds, lines, a, kmeans):
    res = kmeans.predict(a)
    linesC = []
    aC = []
    for i in range(len(res)):
        if res[i] in clusterIds:
            linesC.append(lines[i])
            emb = a[i, :]
            aC.append(emb.reshape(1, (len(emb))))
    aC = numpy.concatenate(aC)
    return (linesC, aC)


# for plotting
SHOW_ALL = 0; SHOW_LOW = 1; SHOW_MIDDLE = 2; SHOW_HIGH = 3

def getScaledSizes(a, kmeans, sort_order, accumulate=True):
    sizes = get_cluster_sizes(a, kmeans)
    sizes = [sizes[i] for i in sort_order]  # use sort_order order
    scale = numpy.sum(sizes)
    res = []
    res.append(sizes[0] / scale)
    for i in range(1, len(sizes)):
        res.append(sizes[i] / scale)  # size scaled to sum to 1
        if accumulate:
            res[-1] += res[-2]  # add previous value
    return res

def getXRange(show, low, middle, high):
    x = []
    if show == SHOW_ALL:
        x = range(high)
    elif show == SHOW_LOW:
        x = range(low)
    elif show == SHOW_MIDDLE:
        x = range(low, middle)
    else:
        x = range(middle, high)
    return x

