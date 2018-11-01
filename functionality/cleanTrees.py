# -*- coding: utf-8 -*-
"""

Created on January 6, 2018

@author:  neerbek
"""

import sys
sys.setrecursionlimit(10000)  # for large trees
# os.chdir("..")
from numpy.random import RandomState  # type: ignore

import ai_util
import similarity.load_trees as load_trees


inputtrees = None
outputtrees = None
trees = None
max_tree_count = -1
sentenceCutoffLow = -1
sentenceCutoffHigh = -1
doShuffle = False
randomSeed = 1234
totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Eval time: ")
totaltimer.begin()

def syntax():
    print("""syntax: cleanTrees.py [-inputtrees <file>] [-outputtrees <file>] [-sentenceCutoffLow  <int>] [-sentenceCutoffHigh  <int>] [-doShuffle] [-randomSeed <int>]
    [-h | --help | -?]
""")
    sys.exit()


arglist = sys.argv
# arglist = "cleanTrees -inputtrees ../taboo-jan/functionality/201/train_full_random.txt -sentenceCutoffLow 5 -sentenceCutoffHigh 200 -outputtrees ../taboo-jan/functionality/201/train_full_random_cleaned.txt".split(" ")
# arglist = "cleanTrees -inputtrees ../taboo-jan/functionality/201/dev_full_random.txt -sentenceCutoffLow 5 -sentenceCutoffHigh 200 -outputtrees ../taboo-jan/functionality/201/dev_full_random_cleaned.txt".split(" ")
# arglist = "cleanTrees -inputtrees ../taboo-jan/functionality/201/test_full_random.txt -sentenceCutoffLow 5 -sentenceCutoffHigh 200 -outputtrees ../taboo-jan/functionality/201/test_full_random_cleaned.txt".split(" ")
# arglist = "cleanTrees -inputtrees ../taboo-jan/functionality/202/data_full_random.txt -sentenceCutoffLow 5 -sentenceCutoffHigh 200 -outputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.txt".split(" ")
# arglist = "cleanTrees -inputtrees ../taboo-jan/functionality/203/data_full.txt -sentenceCutoffLow 5 -sentenceCutoffHigh 200 -doShuffle -randomSeed 83634 -outputtrees ../taboo-jan/functionality/202/data_full_random_cleaned.txt".split(" ")
argn = len(arglist)

i = 1
if argn == 1:
    syntax()

print("Parsing args")
while i < argn:
    setting = arglist[i]
    arg = None
    if i < argn - 1:
        arg = arglist[i + 1]
    # print(setting, arg)

    next_i = i + 2
    if setting == '-inputtrees':
        inputtrees = arg
    elif setting == '-outputtrees':
        outputtrees = arg
    elif setting == '-sentenceCutoffLow':
        sentenceCutoffLow = int(arg)
    elif setting == '-sentenceCutoffHigh':
        sentenceCutoffHigh = int(arg)
    elif setting == '-randomSeed':
        randomSeed = int(arg)
    else:
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        elif setting == "-doShuffle":
            doShuffle = True
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i

if inputtrees == None:
    raise Exception("Need a set of trees on which to train!")

print("loading " + inputtrees)
trees = load_trees.get_trees(file=inputtrees, max_count=max_tree_count)

if sentenceCutoffLow > -1 or sentenceCutoffHigh > -1:
    trees = load_trees.cleanTreesByLength(trees, sentenceCutoffLow, sentenceCutoffHigh)
trees = load_trees.cleanTreesByBadChars(trees)

cleaner = load_trees.SentenceCounter(trees, ["0", "4"])
trees = cleaner.cleanAmbigous(trees)

if doShuffle:
    print("Shuffling")
    rng = RandomState(randomSeed)
    trees = ai_util.shuffleList(trees, rng=rng)

if outputtrees != None:
    print("saving cleaned trees to " + outputtrees)
    load_trees.put_trees(outputtrees, trees)

traintimer.end()

# Done
traintimer.report()
totaltimer.end().report()
