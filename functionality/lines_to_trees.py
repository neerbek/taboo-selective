# -*- coding: utf-8 -*-
"""

Created on December 8, 2017

@author:  neerbek
"""

import os

import confusion_matrix
import similarity.load_trees as load_trees

# But trees already exists as kmeans_treesC2.txt and kmeans_trees2C2.txt

core_dir = "/home/neerbek/jan/phd/DLP/paraphrase/taboo-core/"
inputfile = "output/kmeans_embeddingsC2.txt"
outputfile = "output/kmeans_embeddingsC2_only_trees.txt"
lines = confusion_matrix.read_embeddings(os.path.join(core_dir, inputfile))
trees = [line.tree for line in lines]
load_trees.put_trees(os.path.join(core_dir, outputfile), trees)

inputfile_dev = "output/kmeans_embeddings2C2.txt"
outputfile_dev = "output/kmeans_embeddings2C2_only_trees.txt"
lines = confusion_matrix.read_embeddings(os.path.join(core_dir, inputfile_dev))
trees = [line.tree for line in lines]
load_trees.put_trees(os.path.join(core_dir, outputfile_dev), trees)
