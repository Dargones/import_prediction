# Java Local Import Prediction with Gated Graph Neural Networks

## Data Preprocessing and Baselines

This directory contains several ipython notebooks that can be used to preprocess 
the data downloaded from BigQuery and parsed with JavaParser. You might need 
the following packages to run this code: `pytorch, numpy, tqdm, joblib, 
networkx, sklearn, matplotlib, pandas`. Below is the breakdown of what each of 
the notebooks is for:

- Filtering.ipynb - contains code that can be used to filter out repositories 
that are duplicates of other repositories, repositories that contain files that
could not be parsed, repositories that contain files that do not conform with 
java package naming conventions, etc. 

- Embeddings.ipynb - defines a simple autoencoder that can be used to reduce the
 dimensionality of glove embeddings so that they can be successfully used to
 create filename embeddings in ConvertingToGraphs.ipynb

- ConvertingToGraphs.ipynb - contains code that allows to convert data generated
 by Filterings.ipynb to a format that could serve as input to a GGNN. Also shows
 some statistics and examples of how such graphs might look like
 
- Baselines.ipynb - defines a set of relatively simple baselines that the GGNN
 can be compared to in terms of performance on the input prediction task. These
 include embedding similarity, node degree, shortest path from one node to 
 another in a graph, etc.
 
- GGNNs.ipynb - notebook for running GGNNs. The model definition, loss function,
and data loader anre all defined in ../python folder. The details of my
implementation of GGNNs are also located there in the README.md file. 
The notebook only gives an example of how my GGNN can be run and stores some
results