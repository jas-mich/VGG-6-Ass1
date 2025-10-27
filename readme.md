# VGG6 for Cifar 10 datasets using hyperparameter sweep to find best model

This project implements a VGG-6 model for image classification on the CIFAR-10 dataset. It uses a hyperparameter sweep to find the best performing model.

## Setup Instructions

**Note:** I developed and tested this project using a GPU. Therefore, I installed PyTorch using Conda to ensure proper CUDA support.

### 1. Create Conda Environment

First, create a Conda environment with Python 3.10.

```bash
conda create --name vgg6 python=3.10 ipython
```

### 2. Activate the Environment

Activate the newly created environment.

```bash
conda activate vgg6
```

### 3. Install PyTorch

PyTorch needs to be installed via Conda to ensure compatibility, especially if you are using a GPU.

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 4. Install Requirements

Install the remaining packages using pip.

```bash
pip install -r requirements.txt
```

**Note:** The `torch` package in `requirements.txt` was not installed using `pip`. It was installed via `conda` as specified in step 3. The `requirements.txt` file is provided for reference for the other packages.

## Training

The model training was initiated using Weights & Biases (`wandb`) sweeps for hyperparameter optimization. Two different search strategies were explored:

1.  **Grid Search:** An exhaustive search over the hyperparameter space, configured in `sweep_config_grid.yaml`. I explored this and with 163+ runs,I fixed the batch size and epochs from this run and followed by subsequent runs using Bayesian Optimization.Have included this file for reference.
2.  **Bayesian Optimization:** A more efficient search strategy that uses the results from previous iterations to inform the next choice of hyperparameters. The training was ultimately performed using this method, with the configuration specified in `sweep_config_bayes.yaml`.

To start a sweep, run the following command:
```bash
wandb sweep <config_file.yaml>
```
For example, to use the Bayesian optimization configuration:
```bash
wandb sweep sweep_config_bayes.yaml
```

This will output a sweep ID. To run the agent, use the following command:

```bash
wandb agent <sweep_id>
```

## Reproducing Results

To reproduce the results and test the best model, you can use the `test.py` script. This script loads the model with its best weights and configuration, which achieved a test accuracy of 88.78%.

To run the script:
```bash
python test.py
```

### Sample Output
```
Files already downloaded and verified
Files already downloaded and verified
Best accuracy model loaded
device: cuda
Evaluating: 100%|███████████████████████████████████████████████████████████████████████| 20/20 [00:07<00:00,  2.71it/s]
Accuracy: 88.78
Loss: 7.197296142578125
```

## Note on Model Checkpoints

Please note that this code was run on a GPU machine. During training (`train.py`), the best performing models were saved to an `./outputs` directory using the following logic:

```python
if test_acc > best_test_acc:
    test_acc_str = str(test_acc).replace(".","_")
    save_path=f"./outputs/bestmodel_{str(lr)}_{test_acc_str}_epoch{str(epoch)}.pth"
```

For this repository, the `/outputs` directory has been ignored. The best model checkpoint has been placed directly in the root directory, and the `test.py` script is configured to load the model from there.
