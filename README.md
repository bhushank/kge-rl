# [Analysis of the Impact of Negative Sampling on Link Prediction in Knowledge Graphs](https://arxiv.org/abs/1708.06816)


This code implements the state-of-the-art Knowledge Graph Embedding [algorithms](http://www.cs.technion.ac.il/~gabr/publications/papers/Nickel2016RRM.pdf) such as [RESCAL](http://www.dbs.ifi.lmu.de/~tresp/papers/p271.pdf), [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data), [COMPLEX](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828), Computational Graphs are implemented using [PyTorch](https://pytorch.org/).

# Installation
## Data
* Create a data directory and download FB15K datasets from [here](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz)
* Rename freebase_mtr100_mte100-train.txt, freebase_mtr100_mte100-valid.txt, freebase_mtr100_mte100-test.txt as train, dev, test respectively. These files should reside inside the data directory.

## Configuration
* Create directory ./data/experiment_specs/
* Experiment configurations are specified as JSON files inside the experiment_specs directory.
* An example configuration file (complex.json) can be found in the repository.
### Specifications
* "batch_size": Train Batch Size,
* "entity_dim": Embedding Dimension (must be specified),
* "exp_name": Experiment Name,
* "is_dev": True if you want to test on validation data (must be specified),
* "is_typed": True if you want to use Type Regularizer (default False),
* "l2": Strength of L2 regularizer (must be specified),
* “lr”: Learning rate for ADAM SGD optimizer
* "model": Model (rescal, transE, distmult, complex) (must be specified),
* "num_epochs": Max number of epochs (default 100),
* “neg_sampler”: Negative Sampling (random, corrupt, relational, typed, nn, adversarial). Must be specified 
* “num_negs”:  Number of negative samples (must be specified).

Additional specifications (early stopping, save and report time, etc) can be changed by modifying constants.py. Default values can also be found in constants.py

# Usage
To train and test simply run  
*python experiment_runner.py "data_dir" "experiment_name"*  
where experiment_name is the name of the JSON file without the .json extension. For example  
*python experiment_runner.py ./data/ freebase_bilinear*

* Additional options: -r to resume training. -v for saving embeddings and -t for fine tuning.
* Note that nearest neighbor and adversarial sampling requires a trained RESCAL model for negative sampler. To use nn, or adversarial, change options in json file and add *-t* to the command line. For example to tune a pre trained complex model execute the following command. Note that complex directory must contain the pretrained parameters in pytorch format.

*python experiment_runner.py ./data/ complex -t*







