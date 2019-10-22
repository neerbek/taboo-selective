# -*- coding: utf-8 -*-
"""

Created on May 18, 2018

@author:  neerbek
"""

import os
from numpy.random import RandomState

import rnn_enron
import similarity.load_trees as load_trees

from rnn_enron import NodeCounter


# copied from rnn_enron
def updateWordVectors(node, wordEmbMap, nodeCounter):
    # assumes is_binary and has_only_words_at_leafs
    nodeCounter.incNode()
    if node.is_leaf():
        nodeCounter.incWord()
        word = node.word.lower()
        # word = node.word
        if word[-1] == ',' and len(word) > 1:
            word = word[:-1]  # remove trailing ,
        if word == "-lrb-":
            word = "("
        elif word == "-rrb-":
            word = ")"
        elif word == "-lsb-":
            word = "("
        elif word == "-rsb-":
            word = ")"
        elif word == "-lcb-":
            word = "("
        elif word == "-rcb-":
            word = ")"
        elif word == "-amp-":
            word = "&"

        if word in wordEmbMap:
            node.representation = wordEmbMap[word].number
        else:
            nodeCounter.incUnknown()
            node.representation = wordEmbMap[rnn_enron.UNKNOWN_WORD].number
            # if nodeCounter.unknown_count<100:
            #    print("unknown word: \"" + word + "\"")
    else:
        updateWordVectors(node.left, wordEmbMap, nodeCounter)
        updateWordVectors(node.right, wordEmbMap, nodeCounter)


# copied from rnn_enron
def initializeTrees(trees, wordEmbMap):
    totalCounter = NodeCounter()
    for tree in trees:
        nodeCounter = NodeCounter()
        updateWordVectors(tree, wordEmbMap, nodeCounter)
        totalCounter.add(nodeCounter)
        label = 0
        if tree.syntax == "1":
            label = 1
        elif tree.syntax == "0":
            label = 0
        else:
            raise Exception("tree does not have correct syntax label: {}".format(tree.syntax))
        rnn_enron.setTreeLabel(tree, label)
    print(
        "Done with tree. Saw {} nodes, {} words and {} unknowns. Unknown ratio is {}".
        format(totalCounter.node_count, totalCounter.word_count,
               totalCounter.unknown_count, totalCounter.getRatio()))


class WordEmb:
    def __init__(self, number, word, representation):
        self.number = number
        self.word = word
        self.representation = representation


def getListOfWordRepresentations(t, res):
    if t is None:
        return
    if t.is_leaf():
        res.append(t.representation)
    else:
        getListOfWordRepresentations(t.left, res)
        getListOfWordRepresentations(t.right, res)


def treesToLSTMFormat(trees, sortByLen, maxLen):
    res_x = []  # type: List[List[float]]
    res_y = []  # type: List[int]
    for t in trees:
        x = []
        getListOfWordRepresentations(t, x)
        y = t.label
        if not maxLen or len(x) < maxLen:
            res_x.append(x)
            res_y.append(y)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sortByLen:
        sorted_index = len_argsort(res_x)
        res_x = [res_x[i] for i in sorted_index]
        res_y = [res_y[i] for i in sorted_index]

    return (res_x, res_y)


def loadEmbeddings(glovePath, nWords, nx):
    rng = RandomState(1234)  # for unknown emb

    LT = rnn_enron.get_word_embeddings(os.path.join(glovePath, "glove.6B.{}d.txt".format(nx)), rng, nWords - 1)
    wordEmbMap = {}
    words = LT.keys()
    # words = sorted(words)  # same numbering every time
    for idx, w in enumerate(words):
        wordEmb = WordEmb(idx, w, LT[w])
        wordEmbMap[w] = wordEmb
    return wordEmbMap


class MonsantoData:
    def __init__(self, path, wordEmbMap, useTestTrees=False, manualSensitive=False):
        self.path = path
        self.wordEmbMap = wordEmbMap
        self.trainTreeName = "$train.txt"
        self.testTreeName = "$dev.txt"
        if manualSensitive:
            self.trainTreeName = "$train_manual_sensitive.txt"
            self.testTreeName = "$dev_manual_sensitive.txt"
        if useTestTrees:
            print("Using real test trees for test")
            self.testTreeName = "$test.txt"
            if manualSensitive:
                self.testTreeName = "$test_manual_sensitive.txt"
        else:
            print("Using our DEV trees for test (and fake dev trees for lstm dev)")
        # Why: because LSTM internally look at the dev set and thus " cheat"/"pollute" the dev set
        # Thus our dev set is more like a test set and we use it as such
        # However we need to evaluate the models on the REAL test also, hence the flag useTestTrees

    def loadData(self, path=None,
                 n_words=100000, valid_portion=0.1, maxlen=None, sortByLen=False):
        validPortion = valid_portion
        maxLen = maxlen
        train_trees = load_trees.get_trees(file=self.path + self.trainTreeName, max_count=-1)
        ratio = int(validPortion * len(train_trees))
        valid_trees = train_trees[:ratio]
        train_trees = train_trees[ratio:]
        test_trees = load_trees.get_trees(file=self.path + self.testTreeName, max_count=-1)

        initializeTrees(train_trees, self.wordEmbMap)
        initializeTrees(valid_trees, self.wordEmbMap)
        initializeTrees(test_trees, self.wordEmbMap)

        (train_set_x, train_set_y) = treesToLSTMFormat(train_trees, sortByLen, maxLen)
        (valid_set_x, valid_set_y) = treesToLSTMFormat(valid_trees, sortByLen, maxLen)
        (test_set_x, test_set_y) = treesToLSTMFormat(test_trees, sortByLen, maxLen)

        train = (train_set_x, train_set_y)
        valid = (valid_set_x, valid_set_y)
        test = (test_set_x, test_set_y)

        return train, valid, test
