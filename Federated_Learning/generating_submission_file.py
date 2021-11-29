"""
With this scrip we generate the submission file as outlined by the ITU challenge
"""

import csv
import os
import time
import pandas as pd
import numpy as np

from federated_learning_agent import *
from preparing_data import *

# Input split as suggested by Andrea
INDEX_COL_FIRST_INPUT = [0, 1, 2, 3]  # RSSI, SINR, distance, OBSS-PD,
INDEX_COL_SECOND_INPUT = range(4, 11)  # 'Nsta', '1stInterf', '2nInterf', '3rdInterf', '4thInterf', '5thInterf', 'num Interf APs'

# To load the trained weights
checkpoint_path = "trained_anns/final_model"  # Write the name of ANN
checkpoint_dir = os.path.dirname(checkpoint_path)

if __name__ == '__main__':
    print('Starting the ITU-ML5G-PS-004: Federated Learning for Spatial Reuse in a multi-BSS (Basic Service Set)')
    print('Generating submission file')
    time_start = time.time()

    # read the test data
    dataset_test = pd.read_pickle('cleaned_dataset/test.pkl')
    # dataset_test = dataset_test.head(5)
    # initialize global model
    # smlp_global = DNN_model()
    smlp_global = Andreas_model()
    global_model = smlp_global.build()
    global_model.load_weights(checkpoint_path)  # Load the weights as we trained by other script
    predictions = []
    for index, row in dataset_test.iterrows():
        #print(row['context'])
        X_input_data = get_test_data([row['context']], dataset_test)
        X_input_data_1 = X_input_data[:, INDEX_COL_FIRST_INPUT]  # The first part of the input vector
        X_input_data_2 = X_input_data[:, INDEX_COL_SECOND_INPUT]  # The second part of the input vector
        row_predictions = global_model.predict([X_input_data_1, X_input_data_2]) * 6.0
        predictions.append([x[0] for x in row_predictions])  # reshape the predictions into a single line...
    print(predictions)
    # Save the file with predictions and down load it
    #predictions_df = pd.DataFrame(predictions)
    #print(predictions_df)
    #predictions_df.to_csv('results/predictions.txt', index=False, header=False)

    with open('results/FederationS_final.csv', "w") as f:
        wr = csv.writer(f)
        wr.writerows(predictions)

    time_end = time.time()
    time_total = time_end - time_start
    print("Simulation Ended and lasted for ", time_total, " s")

