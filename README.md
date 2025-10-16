# MAP-ETNN
This is the repository for the MAP-ETNN model as described and used in the Masters Thesis [MAP-ETNN: Topological Deep
Learning for Protein–Ligand Binding
Affinity]()

This model is based on the E(n) equivariant topological neural network from Battiloro et al. The original codebase can be found [here](https://github.com/NSAPH-Projects/topological-equivariant-networks) and the paper can be found [here](https://arxiv.org/abs/2405.15429). The changes added in this repo consist of:
- Addition of BindingNet dataset and liftings
    - Found in `/conf/conf_bindingnet` and `/etnn/bindingnet`
- Addition of PDBBind/CASF2016 dataset and liftings
    - Found in `/conf/conf_pdb` and `etnn/pdbbind`
- Modification of Global pooling to use multiple aggregation forms (Sum, Max, Mean, Attention)
    - Found in `etnn/model_head.py`


## Training and Evaluation
Training and evaluation are combined in a single program which can be run as experiments, these files follow the naming structure `main_*.py` and can be found in the top-level folder. Each experiment uses a [Hydra]() configuration, all of which are stored in the `conf` folder. By default each experiment will use the `config.yaml` found in each datasets respective folder. The dataset and experiment configurations can be passed in the CLI as needed as well as any configuration changes that need to be overriden, I recommend looking at the [Hydra documentation]() for a quick introduction.

## Data Processing
The data is created using a Pytorch Geometric InMemoryDataset which processes the dataset before training and stores it as a `.pt` file. This has the benefit of preprocessing once and then reusing the stored data. If changes are made to the way the data is preprocessed the dataset can also be forced to reload the dataset in the config files. 