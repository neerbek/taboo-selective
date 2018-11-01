# -*- coding: utf-8 -*-
"""

Created on November 6, 2017

@author:  neerbek

Reads a bunch of flat models and set of embeddings and runs the models and outputs
a concatenated set of embeddings

Can only take hardcoded architectures for now
"""
import sys
from typing import List
import numpy  # type: ignore
from numpy.random import RandomState  # type: ignore

import jan_ai_util
import confusion_matrix
import rnn_model.FlatTrainer

# import importlib
# importlib.reload(kutil)
inputfile = "../../taboo-core/output_embeddings.txt"

outputfile = "multioutput.txt"
inputmodels: List[str]; inputmodels = []
randomseeds: List[int]; randomseeds = []

timers = jan_ai_util.Timers()


def syntax():
    print("""syntax: run_multi_flat_models.py
    -inputfile <filename> |
    -inputmodels <filename>[,<filename>,...] | -randomseeds <int>[,<int>,...]
    -outputfile <string>
    [-h | --help | -?]

-inputfile is a list of final sentence embeddings in the format of run_model_verbose.py
-inputmodels is a comma seperated list of models to run on the embeddings
-randomseeds is a comma seperated list of seeds for featuredrop (which is per model)
-outputfile is the filename of where to save output
""")
    sys.exit()


# OMP_NUM_THREADS=2 ipython3 ../taboo-jan/functionality/run_multi_flat_models.py -- -inputfile output/kmeans_embeddings2C2.txt -inputmodels ../taboo-jan/functionality/logs/save_exp97a.zip\$save_exp97a_best.txt,../taboo-jan/functionality/logs/save_exp97b.zip\$save_exp97b_best.txt -randomseeds 37624,434
arglist = sys.argv
# arglist = ...
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

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == '-inputfile':
        inputfile = arg
    elif setting == '-inputmodels':
        inputmodels = arg.split(",")
    elif setting == '-randomseeds':
        tmp = arg.split(",")
        randomseeds = [int(f) for f in tmp]
    elif setting == '-outputfile':
        outputfile = arg
    else:
        # expected option with no argument
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


lines = confusion_matrix.read_embeddings(inputfile, max_line_count=-1)

numberOfHiddenLayers = 2
retain_probability = 0.9
hiddenLayerSize = 100
featureDropCount = 15
nIn = len(lines[0].emb) + 1 - featureDropCount
batchSize = 300

def buildModel():
    isDropoutEnabled = False
    rng = RandomState(2)
    model = rnn_model.FlatTrainer.RNNContainer(nIn=nIn, isDropoutEnabled=isDropoutEnabled, rng=rng)
    for i in range(numberOfHiddenLayers):
        dropout = rnn_model.FlatTrainer.DropoutLayer(model, retain_probability, rnn_model.FlatTrainer.ReluLayer(nOut=hiddenLayerSize))
        model.addLayer(dropout)
    model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=2))
    return model


result: List[List[float]]; result = [[] for l in lines]
for i in range(len(inputmodels)):
    inputmodel = inputmodels[i]
    randomseed = randomseeds[i]
    a1 = confusion_matrix.get_embedding_matrix(lines, normalize=True)
    rng = RandomState(randomseed)
    perm = rng.permutation(a1.shape[1])
    featureDropArray = perm[:featureDropCount]
    for f in featureDropArray:
        a1[:, f] = 0
    trainX = jan_ai_util.addBiasColumn(a1)
    model = buildModel()
    model.load(inputmodel)
    # get final embeddings

    trainParam = rnn_model.FlatTrainer.TrainParam()
    modelEvaluator = rnn_model.FlatTrainer.ModelEvaluator(model, trainParam)

    n = trainX.shape[0]
    if batchSize == 0:
        batchSize = n
    nBatches = int(numpy.ceil(n / batchSize))
    emb = None
    index = 0
    for j in range(nBatches):
        xIn = trainX[j * batchSize: (j + 1) * batchSize]
        emb = modelEvaluator.getEmbedding(xIn)
        for k in range(emb.shape[0]):
            e = []
            for m in range(emb.shape[1]):
                e.append(emb[k, m])
            result[index].extend(e)
            index += 1
    print("shape is :", i, emb.shape)


exp = len(result[0])
print("length is: ", exp)
for i in range(len(result)):
    if len(result[i]) != exp:
        raise Exception("Length mismatch at index {}. Expected {} got {}".format(i, exp, len(result[i])))
    lines[i].emb = result[i]
confusion_matrix.write_embeddings(outputfile, lines)
timers.endAndReport()
