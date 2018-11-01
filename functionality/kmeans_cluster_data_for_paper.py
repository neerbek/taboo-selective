import pylab  # type: ignore
from numpy.random import RandomState  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
import numpy  # type: ignore

import confusion_matrix

# ## Debug values, replace with actuals
a = numpy.zeros(shape=(10, 100))
a2 = numpy.zeros(shape=(10, 100))
lines = None
lines2 = None

# ################################################################################
#
# ## dist of label counts
#
def get_cluster_counts(emb_matrix, lines, n_clusters, rng):
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng).fit(emb_matrix)
    res = kmeans.predict(emb_matrix)
    label_count_sen = [0 for i in range(len(kmeans.cluster_centers_))]
    label_count_both = [0 for i in range(len(kmeans.cluster_centers_))]
    for i in range(len(res)):
        l = res[i]
        line = lines[i]
        if line.ground_truth == 4:
            label_count_sen[l] += 1
        label_count_both[l] += 1
    return sorted(label_count_both)


n_clusters = 35
y1 = get_cluster_counts(a, lines, n_clusters, RandomState(2))
y2 = get_cluster_counts(a, lines, n_clusters, RandomState(387))
y3 = get_cluster_counts(a, lines, n_clusters, RandomState(910))
y4 = get_cluster_counts(a, lines, n_clusters, RandomState(2094))
y5 = get_cluster_counts(a, lines, n_clusters, RandomState(48974))
y6 = [(y1[i] + y2[i] + y3[i] + y4[i] + y5[i]) / 5 for i in range(len(y1))]


x = range(len(y1))

confusion_matrix.new_graph('Clusters', 'Counts')
pylab.plot(x, y1, 'g:', label='run $1$')
pylab.plot(x, y2, 'm:', label='run $2$')
pylab.plot(x, y3, 'k:', label='run $3$')
pylab.plot(x, y4, 'r:', label='run $4$')
pylab.plot(x, y5, 'y:', label='run $5$')
pylab.plot(x, y6, '-k', label='average')
pylab.legend()
pylab.show()
# pylab.savefig('../../../ExperimentDescription/figures/kmeans_counts_35' + '.eps')


# ################################################################################
#
# ## dist of label ratios
#
def stable_div(a, b):
    if b == 0:
        return 0
    return a / b

def get_cluster_ratios_sort_order(emb_matrix, lines, kmeans):
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
    ratio = [stable_div(label_count_both[i], emb_matrix.shape[0]) for i in range(len(label_count_both))]
    ratio = numpy.array(ratio)
    indexes = numpy.argsort(ratio)  # cluster_id to sort_index, asc
    return [i for i in indexes]

def get_cluster_ratios(emb_matrix, lines, kmeans, sort_order):
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
    ratio = [stable_div(label_count_both[i], emb_matrix.shape[0]) for i in range(len(label_count_both))]
    return [ratio[i] for i in sort_order]  # return in sort_order order


kmeans = KMeans(n_clusters=35, random_state=RandomState(2)).fit(a)
sort_order = get_cluster_ratios_sort_order(a, lines, kmeans)
y1 = get_cluster_ratios(a, lines, kmeans, sort_order)
prev = 0
for y in y1:
    if y < prev:
        print("sorting is wrong for {}, {}".format(y, prev))
    prev = y
y2 = get_cluster_ratios(a2, lines2, kmeans, sort_order)

kmeans = KMeans(n_clusters=35, random_state=RandomState(387)).fit(a)
sort_order = get_cluster_ratios_sort_order(a, lines, kmeans)
y3 = get_cluster_ratios(a, lines, kmeans, sort_order)
prev = 0
for y in y3:
    if y < prev:
        print("sorting is wrong for {}, {}".format(y, prev))
    prev = y
y4 = get_cluster_ratios(a2, lines2, kmeans, sort_order)

kmeans = KMeans(n_clusters=35, random_state=RandomState(910)).fit(a)
sort_order = get_cluster_ratios_sort_order(a, lines, kmeans)
y5 = get_cluster_ratios(a, lines, kmeans, sort_order)
prev = 0
for y in y5:
    if y < prev:
        print("sorting is wrong for {}, {}".format(y, prev))
    prev = y
y6 = get_cluster_ratios(a2, lines2, kmeans, sort_order)

kmeans = KMeans(n_clusters=35, random_state=RandomState(2094)).fit(a)
sort_order = get_cluster_ratios_sort_order(a, lines, kmeans)
y7 = get_cluster_ratios(a, lines, kmeans, sort_order)
prev = 0
for y in y7:
    if y < prev:
        print("sorting is wrong for {}, {}".format(y, prev))
    prev = y
y8 = get_cluster_ratios(a2, lines2, kmeans, sort_order)

kmeans = KMeans(n_clusters=35, random_state=RandomState(48974)).fit(a)
sort_order = get_cluster_ratios_sort_order(a, lines, kmeans)
y9 = get_cluster_ratios(a, lines, kmeans, sort_order)
prev = 0
for y in y9:
    if y < prev:
        print("sorting is wrong for {}, {}".format(y, prev))
    prev = y
y10 = get_cluster_ratios(a2, lines2, kmeans, sort_order)

y11 = [(y1[i] + y3[i] + y5[i] + y7[i] + y9[i]) / 5 for i in range(len(y1))]
y12 = [(y2[i] + y4[i] + y6[i] + y8[i] + y10[i]) / 5 for i in range(len(y1))]


x = range(len(y1))

confusion_matrix.new_graph('Clusters', 'Size Ratio')
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
# pylab.savefig('../../../ExperimentDescription/figures/kmeans_ratios_35' + '.eps')
