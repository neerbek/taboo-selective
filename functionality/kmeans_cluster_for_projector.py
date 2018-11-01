# -*- coding: utf-8 -*-
"""

Created on October 3, 2017

@author:  neerbek

For google PCA/t-SNE projector: http://projector.tensorflow.org/
"""
import sys
import io
from numpy.random import RandomState  # type: ignore

import ai_util
import rnn_model.rnn
import rnn_model.learn
import rnn_model.FlatTrainer
import confusion_matrix
import similarity.load_trees as load_trees

inputfile = "../../taboo-core/output_embeddings.txt"

trainParam = rnn_model.FlatTrainer.TrainParam()
trainParam.retain_probability = 0.9
trainParam.batchSize = 500
randomSeed = 7485

totaltimer = ai_util.Timer("Total time: ")
traintimer = ai_util.Timer("Train time: ")
totaltimer.begin()


def syntax():
    print("""syntax: kmeans_cluster_for_projector.py
-inputfile <filename> | -randomSeed <int> |
    [-h | --help | -?]

-inputfile is a list of final sentence embeddings in the format of run_model_verbose.py
-randomSeed initialize the random number generator
""")
    sys.exit()


arglist = sys.argv
# arglist = "kmeans_cluster_for_projector.py -inputfile ../../taboo-core/output/output_embeddings.txt".split(" ")
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
    elif setting == '-randomSeed':
        randomSeed = int(arg)
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

rng = RandomState(randomSeed)
a = confusion_matrix.get_embedding_matrix(lines, normalize=True)
confusion_matrix.verify_matrix_normalized(a)

maxLineCount = -1
count = 0
with io.open("jan1.txt", 'w', encoding='utf8') as f1:
    with io.open("jan2.txt", 'w', encoding='utf8') as f2:
        count += 1
        for i in range(len(lines)):
            if count == maxLineCount:
                break
            line = lines[i]
            e = a[i, :]
            arr = ""
            for c in range(a.shape[1]):
                arr += str(e[c]) + "\t"
            arr = arr[:-1]
            f1.write(arr)
            f1.write("\n")
            text = load_trees.output_sentence(line.tree)
            f2.write(str(i))
            f2.write("\t")
            f2.write(text)
            f2.write("\n")

