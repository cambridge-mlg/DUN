# Depth Uncertainty in Neural Networks

<p align="center">
<img align="middle" src="./dun_training.gif" width="573" height="272" alt="Training a 10 layer DUN on our toy 'Matern' dataset."/>
</p>

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2006.08437-B31B1B.svg)](https://arxiv.org/abs/2006.08437)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![Pytorch 1.3](https://img.shields.io/badge/pytorch-1.3.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cambridge-mlg/arch_uncert/blob/master/LICENSE)

Existing methods for estimating uncertainty in deep learning tend to require multiple forward passes, making them unsuitable for applications where computational resources are limited. To solve this, we perform probabilistic reasoning over the depth of neural networks. Different depths correspond to subnetworks which share weights and whose predictions are combined via marginalisation, yielding model uncertainty. By exploiting the sequential structure of feed-forward networks, we are able to both evaluate our training objective and make predictions *with a single forward pass*. We validate our approach on real-world regression and image classification tasks. Our approach provides uncertainty calibration, robustness to dataset shift, and accuracies competitive with more computationally expensive baselines. 

### Requirements
Python packages:
* hpbandster 0.7.4
* jupyter 1.0.0
* matplotlib 3.2.1
* numpy 1.18.3
* pandas 0.24.2
* Pillow 6.2.0
* test-tube 0.7.5
* torch 1.3.1
* torchvision 0.4.2
* tqdm 4.44.1

## Running Experiments from the Paper

Install the DUN package, and its requirements, by running the following commands in the root directory of the project:

```bash
pip install -r requirements.txt
pip install -e .
```

Change to the `experiments` directory:
```bash
cd experiments
```

### Toy Data Experiments

First change to the `toy` subdirectory:

```bash
cd experiments/toy
```

All experiments with toy data can be produced with the following script. Plots are generated automatically.

```bash
python train_toy.py --help
```

For example:

```bash
python train_toy.py --inference DUN --N_layers 10  --overcount 1  --width 100  --n_epochs 1000 --dataset wiggle --lr 0.001 --wd 0.0001
```

### Regression Experiments

First change to the `regression` subdirectory:

```bash
cd experiments/regression
```

The regression experiments require 4 stages of computation:
1. Hyperparameter optimisation
2. Training models with the optimal hyperparameters
3. Evaluating all models
4. Plotting the results

For stage 1, change to the `hyperparams` subdirectory:

```bash
cd experiments/regression/hyperparams
```

Then run the following script for all combinations of datasets, splits, and inference methods of interest.

```bash
python run_opt.py --help
```

For example:

```bash
python run_opt.py --dataset boston --n_split 0 --min_budget 200 --max_budget 2000 --early_stop 200 --n_iterations 20 --run_id 0 --method SGD 
```

Note that a unique `run_id` must be suplied for each run.

For stage 2, change to the `retrain_best` subdirectory:

```bash
cd experiments/regression/retrain_best
```

Then run the following script for all combinations of datasets, splits, and inference methods for which hyperparameter optimisation was run.

```bash
python final_run.py --help
```

For example:

```bash
python final_train.py --dataset boston --split 0 --method SGD
```

For stage 3, go to the `regression` subdirectory:

```bash
cd experiments/regression
```

Run the following script.

```bash
python evaluate_final_models_unnorm.py --help
```

This script shouldn't require any command line arguments to run, for example:

```bash
python evaluate_final_models_unnorm.py
```

For stage 4, go to the `experiments` directory:

```bash
cd experiments
```

The regression experiment plots can now be generated by executing the appropriate cells in the `regression_and_image_PLOTS.ipynb` notebook. Launch `jupyter`:

```bash
jupyter-notebook
```

Note that the flights dataset must first be un-zipped before it can be used.

### Image Experiments

First change to the `image` subdirectory:

```bash
cd experiments/image
```

The image experiments require 4 stages of computation:
1. Training baselines
2. Training DUNs
3. Evaluate the models
4. Plotting the results

For stage 1, run the `train_baselines` script, for each baseline configuration and dataset of interest:

```bash
python train_baselines.py --help
```

Note that, unlike the toy data and regression experiments, there are no flags to specify the inference method to use. Instead the inference method is implicit. For example to train an SGD model, we do not need to change any of the default arguments:

```bash
python train_baselines.py --dataset MNIST
```

To train a dropout model, just specify they `p_drop` and `mcsamples` arguments:

```bash
python train_baselines.py --dataset MNIST --p_drop 0.1 --mcsamples 10
```

To train an ensemble, simply train multiple SGD models. Each model will automatically be saved to a unique directory. 

```bash
python train_baselines.py --dataset MNIST; 
python train_baselines.py --dataset MNIST; 
python train_baselines.py --dataset MNIST
```

For stage 2, run the `train_DUN` script, for each DUN configuration and dataset of interest:

```bash
python train_DUN.py --help
```

For example:

```bash
python train_DUN.py --dataset MNIST
```

For stage 3, run the `run_final_image_experiments` script for each inference method and dataset trained in the previous steps.

```bash
python run_final_image_experiments.py --help
```

For example:

```bash
python run_final_image_experiments.py --method=DUN --dataset=MNIST
```

or 

```bash
python run_final_image_experiments.py --method=ensemble --dataset=CIFAR10
```

For stage 4, go to the `experiments` directory:

```bash
cd experiments
```

The image experiment plots can now be generated by executing the appropriate cells in the `regression_and_image_PLOTS.ipynb` notebook. Launch `jupyter`:

```bash
jupyter-notebook
```

## Citation

If you find this code useful, please consider citing our paper:

> Javier Antorán, James Urquhart Allingham, & José Miguel Hernández-Lobato. (2020). Depth Uncertainty in Neural Networks. [[bibtex]](DUN.bib)
