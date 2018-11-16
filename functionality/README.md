# Taboo-selective

A library for doing selective training on recurrent and recursive neural networks.

To clone:

```
git clone https://bitbucket.alexandra.dk/scm/tab/taboo-selective.git
```

## Notes:

Depends on taboo-core (and all the dependencies for taboo-core). For plots you also need matplotlib


## Testing installation
To run tests: from taboo-selective/functionality, say:

```
./run_tests.sh
```

## Usage

The purpose of taboo-selective is to minimize the number of training iterations needed. This guide assumes that a model already have been trained by taboo-core/functionality/train_model.sh

***Overview***
* Step 1: Run pretraining (on randomly initialized model). Save model as <pretrain-model>
* Step 2: Get resulting embedding using <pretrain-model> for each sentence in your dataset (train, dev and test)
* Step 3: Cluster train set embeddings, assign these clusters to dev and test set
* Step 4: Train (as normal) on reduced dataset using your faviourite stopping criteria
* Step 5: Compare the total number of minibatches visited in pretraining + normal (reduced) training and compare to standard approach (no pretraining)


### Details

#### Step 1

Normally you would just run pretraining once, but if you are interested in experimentally finding the best cutoff you can run `train_model.sh` with the following arguments to save a pretrained model for every 25K minibatches. 

```
-train_report_frequency 25000  -validation_frequency 25000 -output_running_model 25000 -file_prefix save_my_pretrained_model
```

You can make a nice plot over the training graph showing cutoffs using

```
taboo-selective/functionality/generate_training_graph.py
```

You need to edit the file to point to the correct saved log file from your full training and you need to add which cutoffs you have tried. 


#### Step 2

Select (or generate) a pretraining model from previous step. Then run the following command on all your data (train, dev and test) to get embeddings for each sentences

```
cd taboo-core
OMP_NUM_THREADS=2 python functionality/run_model_verbose.py -inputtrees <your_trees> <your usual parameters> -batch_size 200 -inputmodel <pretrained_model> -output_embeddings -runOnAllNodes > <output_file>
```

Where run parameters should be as you trained with. In the following we assume you used `output_embeddings.txt` as output file.

#### Step 3

Fitting a clustering on your train set of embeddings and apply (predict) the clustering on dev and test set.

```
../taboo-selective/functionality/kmeans-cluster.py
```

You need to edit the first few lines of the script to correspond to your files and paths.

The kmeans-cluster makes the clustering and splits the data. Optionally it can print out a graph over cluster purities with cutoffs shown

After the clustering you may optionally zip the split datasets such that it is easy to reference to them later on

```
zip -m <output-zip> C1.txt C2.txt 2C1.txt 2C2.txt 3C1.txt 3C2.txt
```

C1.txt and C2.txt is train set split by the low cluster cutoff (from `kmeans-cluster.py`). Similarily 2C1 and 2C2 is the split of the dev set and 3C1 and 3C2 is the test split.

The script also outputs the accuracies of each split. This makes it easier to calculate total accuracies in the next step.

#### Step 4

Train on the part of the data where the MFO score is low. I.e. the part of the split which may benefit most from futher training.

This is done by training as normal but this time on the new split

```
train_model.sh <your normal applied parameters> -traintrees <output-zip>$C2.txt -validtrees <output-zip>$2C2.txt -testtrees <output-zip>$3C2.txt
```

Once your model has converged as required you can find the new obtained accuracies for the split by running run_model.

```
run_model.sh <your normal applied parameters> -inputtrees <output-zip>$C2.txt -inputmodel <new model>
```

Run for all 3 datasets C2, 2C2 and 3C3
