# Java Local Import Prediction with Gated Graph Neural Networks

## Motivation behind using GGNNs

The output of any Gated Graph Neural Network is defined in terms of node
embeddings that it computes. These embeddings are updated a fixed number of 
times and ultimately depend on the embeddings of the neighbouring nodes. A
change of the graph structure will therefore lead to a change of the embeddings.
The core idea behind using GGNNs for import prediction is that an addition of a "logical" import statement to a file should lead to smaller changes in the corresponding embedding than the addition of a completely random import statement. 

Given a repository and a set of potential
import statement to be added to a certain file in that repository, I first run 
the network on the repository graph and record the embedding of the file in
question. For every potential import, I then update the graph by adding a
corresponding edge to it and run the network on this new graph recording the
new embedding for the file under consideration. The import option selected as
"correct" corresponds to the embedding most similar to the original one.
 
 This approach is most similar to the one employed by 
 [Miltiadis *et al.*](https://arxiv.org/abs/1711.00740) to select the most 
 likely variable name to be referenced at a given location out of a set of 
 variable names of matching type defined in the same scope. Many of the
 implementation decision that I made were guided by this previous study. Just
 as Miltiadis *et al.* do, for instance, I use a single neural layer to learn
 similarity between embeddings.
 
 ## Implementing GGNNs in pytorch
 
 I decided to use pytorch for this project because I have more experience with pytorch than tensorflow and because the former is much easier to debug, which
 I believe is a major advantage. 
 
 As the basis for my implementation I selected to use [this GitHub repository](https://github.com/chingyaoc/ggnn.pytorch/blob/master/model.py). While I have
 not modified the way node embeddings are computed in any significant way, 
 I completely changed the way these embeddings are used down the road and have
 written a custom loss function suitable to the task (which is essentially a
 custom triple loss function that takes as input an anchor, a single positive 
 and an arbitrary number of negatives and computes the mean triplet loss 
 across all anchor-positive-negative triplets)
 
 ## This directory's contents
 
 - data_loader.py  - loads graphs in .json format as saved by 
 ConvertingToGraphs.ipynb and converts them to pytorch Dataset for easy usage.
 
 - model.py - the definition of GGNN class. This is largely the code from the
 GitHub repository referenced above, although I had to modify the forward pass
 to suit the present project.
 
 - utils.py - currently contains the loss function used to train the network for the import prediction task.
 
 - wrapper.py - an example of how the model could be run