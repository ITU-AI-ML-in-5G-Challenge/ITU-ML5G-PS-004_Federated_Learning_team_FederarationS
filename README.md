# FederationS: Federated Learning for Spatial Reuse in a multi-BSS (Basic Service Set)scenario 

This repository contains the team FederationS solution for the [ITU-ML5G-PS-004](https://www.upf.edu/web/wnrg/2021-edition) challenge. 

## Overview

We are all aware that Wi-Fi's Achilles heel is its poor performance in crowded scenarios such as airports, stadiums and public gatherings. The recent IEEE 802.11ax amendment includes spatial reuse (SR) to increase simultaneous transmissions and improve Wi-Fi performance in these situations. However,
one of the main limitations of the SR mechanism is that it relies on local information, limiting the effectiveness of this technique. The solution in this repository proposes a distributed method based on Federated Learning (FL), which enables the selection of the best Overlapping Basic Service Set (OBSS)-Preamble Detection(PD) configuration by predicting the achievable throughput and without transferring data to a central server. 

The python notebooks contain the data analysis the pre-processing of the synthetic data [provided by ITU](https://zenodo.org/record/5352060#.YZ0Q7NbP23J). Additionally, one python notebook contains a central approach solution we relied on during our design process. The folder "Federated_Learning" holds the solution(main.py), along with the trained neural network model and other support scripts we used.



## Data Analysis, Data Preparation, and Centralised Solution

**Data_analysis.ipynb**: 

**Central_model.ipynb**:

**Preparing_test_dataset.ipynb**:

**Preparing_training_dataset.ipynb**:


## Federated Learning Solution

The folder "Federated_Learning"  contains the following pythons scripts:

**main.py**:

**federated_learning_agent.py**:

**preparing_data.py**:

**generating_submission_file.py**: The script used to generate throughput prediction file(FederationS_final.csv), saved in subfolder “results”. 

**requirements.txt **: Recording of used python packages.

Before running the scripts unzip **cleaned_dataset.zip ** . The neural network we used to generate the predictions is subfolder “trained_anns”.

## Team

[Andrea Bonfante](https://www.linkedin.com/in/andreabonfante/), CONNECT Centre, Trinity College Dublin, Ireland

[Jernej Hribar](https://www.linkedin.com/in/jernej-hribar/), CONNECT Centre, Trinity College Dublin, Ireland