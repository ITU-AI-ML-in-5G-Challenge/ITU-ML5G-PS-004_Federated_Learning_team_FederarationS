# FederationS: Federated Learning for Spatial Reuse in a multi-BSS (Basic Service Set)scenario 

This repository contains the team FederationS solution, report, and presentation for the [ITU-ML5G-PS-004](https://www.upf.edu/web/wnrg/2021-edition) challenge. 

## Overview

We are all aware that Wi-Fi's Achilles heel is its poor performance in crowded scenarios such as airports, stadiums and public gatherings. The recent IEEE 802.11ax amendment includes spatial reuse (SR) to increase simultaneous transmissions and improve Wi-Fi performance in these situations. However,
one of the main limitations of the SR mechanism is that it relies on local information, limiting the effectiveness of this technique. The solution in this repository proposes a distributed method based on Federated Learning (FL), which enables the selection of the best Overlapping Basic Service Set (OBSS)-Preamble Detection(PD) configuration by predicting the achievable throughput and without transferring data to a central server. 

The python notebooks contain the data analysis the pre-processing of the synthetic data [provided by ITU](https://zenodo.org/record/5352060#.YZ0Q7NbP23J). Additionally, one python notebook contains a central approach solution we relied on during our design process. The folder "Federated_Learning" holds the solution(main.py), along with the trained neural network model and other support scripts we used.



## Data Analysis, Data Preparation, and Centralised Solution

**Data_analysis_and_Central_solution.ipynb**: This notebook contains initial data analysis and central solution.

**Preparing_training_dataset.ipynb**:  This notebook generates pickle file containing training dataset(train.pkl). The pickle file is used as an input in our main training script!

**Preparing_test_dataset.ipynb**: This notebook generates pickle file containing test dataset(test.pkl). The pickle file is used as an input in our main training script!


## Federated Learning Solution

The folder "Federated_Learning"  contains the following pythons scripts:

**main.py**: The main training script for the Federated Learning Solution.

**federated_learning_agent.py**: This python file contains support functions for the federated learning agent used in the main.py.

**preparing_data.py**: This python file contains functions related to data preparation used in main.py script.

**generating_submission_file.py**: Python script used to generate throughput prediction file(FederationS_final.csv), saved in subfolder “results”. 

**requirements.txt**: Recording of used python packages.

Before running the scripts unzip **cleaned_dataset.zip** . The neural network we used to generate the predictions is subfolder “trained_anns”.

## Team

[Andrea Bonfante](https://www.linkedin.com/in/andreabonfante/), CONNECT Centre, Trinity College Dublin, Ireland

[Jernej Hribar](https://www.linkedin.com/in/jernej-hribar/), CONNECT Centre, Trinity College Dublin, Ireland