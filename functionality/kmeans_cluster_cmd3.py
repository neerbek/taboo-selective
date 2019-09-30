# -*- coding: utf-8 -*-
"""

Created on October 3, 2017

@author:  neerbek

Trains a RNN flat model on input list of trees and encodings
"""
import sys
from numpy.random import RandomState  # type: ignore

import ai_util
import jan_ai_util
import rnn_model.rnn
import rnn_model.learn
import rnn_model.FlatTrainer
import confusion_matrix
import kmeans_cluster_util as kutil

# import importlib
# importlib.reload(jan_ai_util)
# os.chdir("../../taboo-core")
inputfile = "output/kmeans_embeddingsC1.txt"
inputdevfile = "output/kmeans_embeddings2C1.txt"
extradata = None
runOnly = False

trainParam = rnn_model.FlatTrainer.TrainParam()
trainParam.retain_probability = 0.9
trainParam.batchSize = 500
randomSeed = 7485
hiddenLayerSize = 150
numberOfHiddenLayers = 2
nEpochs = 5 * 128
trainReportFrequency = 32 * 72
validationFrequency = 64 * 72
inputmodel = None
filePrefix = "save"
learnRate = 0.5
momentum = 0.0
featureDropCount = 0

timers = jan_ai_util.Timers()


def syntax():
    print("""syntax: kmeans_cluster_cmd3.py
-inputfile <filename> | -inputdevfile <filename> | -extradata <file> |-retain_probability <float> |
-batchSize <int> | -randomSeed <int> | -hiddenLayerSize <int> | -numberOfHiddenLayers <int> |
-nEpochs <int> | -learnRate <float> | -momentum <float> | -trainReportFrequency <int> |
-validationFrequency <int> | -inputmodel <filename> | filePrefix <string> | -runOnly
-h | --help | -?

-inputfile is a list of final sentence embeddings in the format of run_model_verbose.py
-inputdevfile is a list of final sentence embeddings in the format of run_model_verbose.py
-extradata is a file with node embeddigs for sentences in inputfile. In the format of run_model_verbose.py. Exact match on sentences are used to select which node values to use.
-inputmodel is an optioal previous saved set of parameters for the NN model which will be loaded

-retain_probability the probability of a neuron NOT being dropped in dropout
-batchSize the number of embeddings trained in a minibatch
-randomSeed initialize the random number generator
-hiddenLayerSize number of neurons in the hidden layer(s)
-numberOfHiddenLayers number of hidden layers
-nEpochs number of complete loops of the training data to do
-learnRate - learnrate for gradient (w/o momentum) learner
-momentum - momentum for gradient (with momentum) learner
-L1param - weight of L1 regularization
-L2param - weight of L1 regularization

-featureDropCount - number of random features to drop (set to 0)
-trainReportFrequency - number of minibatches to do before outputting progress on training set
-validationFrequency - number of minibatches to do before outputting progress on validation set
-filePrefix is a prefix added to all saved model parameters in this run
-runOnly do not train only validates
""")
    sys.exit()


arglist = sys.argv
# arglist = "train_flat_feature_dropout.py -retain_probability 0.9 -hiddenLayerSize 150 -numberOfHiddenLayers 3 -filePrefix save -learnRate 0.01 -momentum 0 -trainReportFrequency 450 -validationFrequency 900 -nEpochs 400 -randomSeed 37624".split(" ")
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

    next_i = i + 2   # assume option with argument (increment by 2)
    if setting == '-inputfile':
        inputfile = arg
    elif setting == '-inputdevfile':
        inputdevfile = arg
    elif setting == '-extradata':
        extradata = arg
    elif setting == '-retain_probability':
        trainParam.retain_probability = float(arg)
    elif setting == '-randomSeed':
        randomSeed = int(arg)
    elif setting == '-batchSize':
        trainParam.batchSize = int(arg)
    elif setting == '-hiddenLayerSize':
        hiddenLayerSize = int(arg)
    elif setting == '-numberOfHiddenLayers':
        numberOfHiddenLayers = int(arg)
    elif setting == '-nEpochs':
        nEpochs = int(arg)
    elif setting == '-learnRate':
        learnRate = float(arg)
    elif setting == '-momentum':
        momentum = float(arg)
    elif setting == '-trainReportFrequency':
        trainReportFrequency = ai_util.eval_expr(arg)
    elif setting == '-validationFrequency':
        validationFrequency = ai_util.eval_expr(arg)
    elif setting == '-inputmodel':
        inputmodel = arg
    elif setting == '-filePrefix':
        filePrefix = arg
    elif setting == '-featureDropCount':
        featureDropCount = int(arg)
    elif setting == '-L1param':
        trainParam.L1param = float(arg)
    elif setting == '-L2param':
        trainParam.L2param = float(arg)
    else:
        # expected option with no argument
        if setting == '-help':
            syntax()
        elif setting == '-?':
            syntax()
        elif setting == '-h':
            syntax()
        elif setting == '-runOnly':
            runOnly = True
        else:
            msg = "unknown option: " + setting
            print(msg)
            syntax()
            raise Exception(msg)
        next_i = i + 1
    i = next_i


lines = confusion_matrix.read_embeddings(inputfile, max_line_count=-1)
if extradata != None:
    lines = confusion_matrix.read_embeddings(extradata, max_line_count=-1, originalLines=lines)
print("number of input train lines {}".format(len(lines)))
a1 = confusion_matrix.get_embedding_matrix(lines, normalize=True)
lines2 = confusion_matrix.read_embeddings(inputdevfile, max_line_count=-1)
a2 = confusion_matrix.get_embedding_matrix(lines2, normalize=True)

rng = RandomState(randomSeed)

print(len(lines), len(lines2))
kutil.get_base_accuracy(lines, "train acc").report()

if featureDropCount > 0:
    perm = rng.permutation(a1.shape[1])
    featureDropArray = perm[:featureDropCount]
    for f in featureDropArray:
        a1[:, f] = 0
        a2[:, f] = 0
    print("featureDropArray", featureDropArray)
    # print(a1[10])

trainParam.X = jan_ai_util.addBiasColumn(a1)  # add 1 bias column, not really needed, but ...
trainParam.valX = jan_ai_util.addBiasColumn(a2)  # add 1 bias column, not really needed, but ...


# format y to 2 class "softmax"
trainParam.Y = jan_ai_util.lines2multiclassification(lines, classes=[0, 4])
trainParam.valY = jan_ai_util.lines2multiclassification(lines2, classes=[0, 4])

trainParam.learner = rnn_model.learn.GradientDecentWithMomentumLearner(lr=learnRate, mc=momentum)

inputSize = trainParam.X.shape[1]


def buildModel(isDropoutEnabled, rng=RandomState(randomSeed)):
    model = rnn_model.FlatTrainer.RNNContainer(nIn=inputSize, isDropoutEnabled=isDropoutEnabled, rng=rng)
    for i in range(numberOfHiddenLayers):
        dropout = rnn_model.FlatTrainer.DropoutLayer(model, trainParam.retain_probability, rnn_model.FlatTrainer.ReluLayer(nOut=hiddenLayerSize))
        model.addLayer(dropout)
    model.addLayer(rnn_model.FlatTrainer.RegressionLayer(nOut=2))
    return model


model = buildModel(True)
if inputmodel != None:
    model.load(inputmodel)

validationModel = buildModel(False)

# actual training
timers.traintimer.begin()
rnn_model.FlatTrainer.train(trainParam, model, validationModel, n_epochs=nEpochs, trainReportFrequency=trainReportFrequency, validationFrequency=validationFrequency, file_prefix=filePrefix, rng=rng, runOnly=runOnly)

# done
timers.endAndReport()
