# Federated Learning and Class Imbalances: Project Structure and Usage Guide

This repository contains a comprehensive suite of code and experiments addressing centralised and federated learning approaches to the semantic segmentation of breast lesions in MRI scans. The main focus is on overcoming challenges in federated learning setups, particularly those caused by data heterogeneity across clients.       

- [Link to detailed documentation](https://whimsical-biscuit-b275f6.netlify.app/](https://spiffy-capybara-0f87fd.netlify.app/))

--- 

The project is organised around several major parts, each targeting specific tasks:

### Main Components:

1. **MNIST Classification**  
   Implements both centralised and federated learning for digit classification on the MNIST dataset. Serves as a baseline for testing federated learning strategies.

2. **U-Net Segmentation**  
   Semantic segmentation with the U-Net architecture. Includes data processing, training, and hyperparameter tuning.

3. **U-Net Paper Adaptation**  
   Modular adaptation of the U-Net model from the [reference paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205), with scripts for training, prediction, and visualisation.

4. **Pretrained Model Fine-tuning**  
   Scripts to fine-tune a pretrained U-Net on a downstream task and compare it with other centralised approaches.

5. **Federated Learning: FedAvg**  
   Federated learning experiments using the standard FedAvg algorithm under various heterogeneity setups (normal, statistical, system).

6. **Federated Learning: FedProx**  
   FedProx implementation for improved training stability under heterogeneity. Results are compared against FedAvg and centralised models.

---

### Code Organisation and Structure:

The project follows common software engineering best practices with clear separation of concerns and modularity:


#### **configs/** – Configuration files for all experiments:
- `fedavg_config.py`, `fedprox_config.py`, `mnist_config.py`, `pretrained_config.py`, `unet_config.py`, `unet_paper_config.py`

#### **docs/** – Notes and supplementary materials.

#### **notebooks/** – Evaluation, plotting, and results analysis:
- `dataset_exploration.ipynb` – Sample visualisation and resizing  
- `dataset_reduction.ipynb` – Subsampling dataset for fast testing  
- `federated_results.ipynb` – Comparison of all federated results  
- `first_unet_results.ipynb` – Centralised U-Net + augmentation impact  
- `optuna_results.ipynb` – Optuna hyperparameter analysis  
- `pretrained_results.ipynb` – Evaluation of pretrained U-Net  
- `unet_paper_results.ipynb` – U-Net paper results + augmentation impact

#### **results/** – Final models, training logs, metrics, and visual outputs:
- `pretrained_model/` – Best/final models, evaluation scripts, visual samples  
- `seg_fedavg/` and `seg_fedprox/` – Federated training outputs (per setup: normal, statistical, system heterogeneity)  
  - `normal_2/` – 2 clients, 50/50 data split  
  - `statistical_het_2/` – 2 clients, 70/30 data split  
  - `statistical_het_6/` – 6 clients, 1x50/5x10 data split  
  - `statistical_het_10/` – 10 clients, 2x30/8x5 data split  
  - `system_het_2/` – Unequal training epochs per client  
- `UNet_paper/` – Training logs, mask examples, arhitecture visualisation  
- `UNet_segmentation/` – Centralised results with and without augmentation  

#### **scripts/** – Standalone run scripts for training, evaluation, and federated execution:
- `MNIST_classification/` – Scripts for centralised and federated MNIST experiments (train, client, and server).
- `pretrained_model/` – Scripts for fine-tuning and evaluating pretrained U-Net models.
- `seg_fedavg/` and `seg_fedprox/` – Scripts to launch server and clients for FedAvg and FedProx experiments.
- `UNet_paper/` – Training and inference scripts for the adaptation of the U-Net paper.
- `UNet_segmentation/` – Scripts for Optuna-based search and training of the optimal U-Net configuration. 

#### **src/** – Core implementation, organised by experiment type:
- `MNIST_classification/` – Model, server, and client logic  
- `pretrained_model/` – Training and evaluation code  
- `seg_fedavg/` and `seg_fedprox/` – Model, aggregation strategy, data handling  
- `UNet_paper/` – Full training pipeline: model, metrics, utils, data prep  
- `UNet_segmentation/` – U-Net model, Optuna search, metrics  
- `plot_notebooks.py` – Shared utilities for generating clean plots in notebooks  

#### **tests/** – Unit tests for core modules:
- `fedavg_test.py`, `fedprox_test.py`, `mnist_test.py`, `pretrained_test.py`, `unet_paper_test.py`, `unet_segmentation_test.py`  

#### **requirnments.txt** – Environment requirnments

---

### Environment and project setup:

1. **Clone the repository**:
    ```bash
    git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/mn628.git
    ```
2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Set paths in the config files**:
    If the data paths or other parameters are different from the default, make sure to set them in all `configs/*.py` files.

4. **Run the pytest files**:
    To check core functionalities, run the pytest files:
    ```bash
    cd tests
    pytest -v .
    ```
5. **Run the scripts and notebooks**:
    Follow the instructions below to recreate the results.

---

### Running the Scripts to Reproduce Results:

#### MNIST Classification


1. Adapt the `config/mnist_config.py` script (change the parameters as needed)
2. Run centralised classification: 
`python scripts/MNIST_classification/mnist_centralised.py`
3. Run federated learning approach:
- Start the server:
  ```
  python scripts/MNIST_classification/server.py
  ```
- Launch N clients (in separate terminals):
  ```
  python scripts/MNIST_classification/client.py 0 3  # Example: client 0 out of 3
  ```

#### Pretrained U-Net (Centralised)

1. Update `configs/pretrained_config.py` script
2. Train the model: `python scripts/pretrained_model/train.py`
3. Evaluate the problem: `python scripts/pretrained_model/evaluate.py`


#### Federated U-Net Segmentation (FedAvg and FedProx)

> Replace `PATH` with `seg_fedavg` or `seg_fedprox` depending on the method.

1. Update parameters in `configs/fedavg_config.py` or `configs/fedprox_config.py`  
2. Start the server:
```
python scripts/PATH/server.py
```
3. (If needed) Get your hostname: `hostname -i` and update it in the config file.
4. Launch N clients:
##### ➤ Statistical Heterogeneity
```
python scripts/PATH/main.py 0 2
```
*(client ID, number of clients)*
##### ➤ System Heterogeneity
```
python scripts/PATH/main.py 0 2 1
```
*(client ID, number of clients, number of epochs for that client)*

---

**Note:**  
- Adjust paths and configuration files as needed for your setup.  
- For specific custom splits, modify the logic in `prepare_data.py` under `UNet_segmentation`, `src/seg_fedavg` or `src/seg_fedprox`.

---

### References
- [Link to detailed documentation](https://whimsical-biscuit-b275f6.netlify.app/](https://spiffy-capybara-0f87fd.netlify.app/))
- [DCE-MRI Dataset](https://www.sciencedirect.com/science/article/abs/pii/S0010482523007205)
- [FedProx Paper](https://arxiv.org/abs/1812.06127)
