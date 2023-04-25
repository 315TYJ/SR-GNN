# SR-GNN
This is the PyTorch implementation of the Semantic and Relational-based Graph Neural Network (SR-GNN) for Knowledge Graph Completion task, as described in our paper:

## Dependencies

PyTorch >= 1.9.0 <br>
DGL >= 0.7.2 (for graph neural network implementation) <br>
Hydra >= 1.1.1 (for project configuration)

## Usage
* code/: includes code scripts.
* data/: <br>
  * dataset/FB15k_237/: FB15k-237 dataset resources <br>
  * dataset/WN18RR/: WN18RR dataset resources <br>
  * output/FB15k_237/: model outputs for FB15k-237 dataset <br>
  * output/WN18RR/: model outputs for WN18RR dataset <br>
* config/: We use Hydra toolkit to manage the model hyper-parameters, which can be either stored in YAML file or passed by command-line. More details can be seen in official docs. <br>
  * config.yaml: project configurations and general parameters <br>
  * dataset/FB15k_237.yaml: best hyper-parameters for FB15k-237 dataset <br>
  * dataset/WN18RR.yaml: best hyper-parameters for WN18RR dataset <br>
## Model Training
```Python 
# enter the project directory
cd SR-GNN-main

# FB15k-237
# set the config/config.yaml `project_dir` field to your own project path
cd FB15K-237
python code/run.py

# WN18RR
# set the config/config.yaml `project_dir` field to your own project path\
cd WN18RR
python code/run.py
```
The model takes about 10h for training on a single GPU, and the GPU memory cost is about 13GB for FB15k-237 and 5GB for WN18RR dataset.
