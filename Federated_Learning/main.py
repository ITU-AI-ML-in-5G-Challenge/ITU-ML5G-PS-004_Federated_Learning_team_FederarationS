"""
This is the main script for training the Federated Learning agent
"""
import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from federated_learning_agent import *
from preparing_data import *

CONTEXT_RANGE = 2000  # In our prepared data the context are numbered between 1000  and 3000, with some missing
NUMBER_SELECTED_CONTEXT = 500  # The number of context we use, note that maximum is 2000 - 54 = 1946, but those are removed later
VALIDATION_PERCENTAGE = 0.05  # The percentage of selected context we use in the validation process
LIST_OF_CONTEXT_WITH_NO_DATA = [1004, 1025, 1116, 1124, 1142, 1163, 1185, 1191, 1205, 1208, 1209, 1224, 1236, 1242,
                                1260, 1282, 1295, 1302, 1312, 1318, 1324, 1341, 1348, 1373, 1405, 1407, 1422, 1471,
                                1485, 1503, 1514, 1520, 1590, 1601, 1640, 1670, 1681, 1693, 1730, 1742, 1766, 1771,
                                1792, 1810, 1811, 1818, 1842, 1856, 1880, 1903, 1931, 1943, 1966, 1990]
# We randomly selected five percent of total available context to be used in validation
TOTAL_NUMBER_OF_VALIDATION_CONTEXT = 97
LIST_OF_VALIDATION_CONTEXT = [2865, 1305, 1111, 2369, 1601, 1398, 1857, 2351, 1165, 2652, 1164, 2484, 2352, 2116, 2744,
                              2697, 1581, 2232, 2826, 2452, 1775, 2485, 2758, 1641, 1386, 2674, 1681, 1471, 1160, 2134,
                              2290, 1504, 2495, 2263, 1828, 2393, 1624, 2242, 2303, 2814, 1562, 1503, 1082, 1434, 1818,
                              2149, 2346, 1350, 2698, 2962, 1136, 2593, 2562, 1208, 1511, 1907, 1515, 1820, 2813, 1885,
                              1343, 2062, 2187, 2522, 2959, 2300, 1548, 2075, 2066, 2516, 2492, 2195, 1260, 1655, 2723,
                              1791, 1786, 1120, 2631, 2969, 1839, 1943, 1349, 2685, 1906, 2540, 2016, 1216, 2883, 1004,
                              2811, 1619, 1175, 1946, 1131, 2428, 2268]
LIST_OF_VALIDATION_CONTEXT_2APS = []
# Federated Learning parameters
BATCH_SIZE = 21  # The size of the batch as used by the context
COMMS_ROUND = 10  # number of communication rounds we use
LR = 0.01
LOSS = 'categorical_crossentropy'
LOAD_WEIGHTS = True  # Set to False if want random init at the start
# Input split as suggested by Andrea
INDEX_COL_FIRST_INPUT = [0, 1, 2, 3]  # RSSI, SINR, distance, OBSS-PD,
INDEX_COL_SECOND_INPUT = range(4, 11)  # 'Nsta', '1stInterf', '2nInterf', '3rdInterf', '4thInterf', '5thInterf', 'num Interf APs'

# Load the saved trained weights
load_checkpoint_path = "trained_anns/saved_model"
load_checkpoint_dir = os.path.dirname(load_checkpoint_path)

# To save the trained weights
checkpoint_path = "trained_anns/final_model"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Multiple cores to be used during ANN traning

if __name__ == '__main__':
    print('Starting the ITU-ML5G-PS-004: Federated Learning for Spatial Reuse in a multi-BSS (Basic Service Set)')
    print('The tensorflow versions we use is:', tf.__version__)
    time_start = time.time()
    # First we read the pre-prepared data from the .csv file
    dataset_train = pd.read_pickle('cleaned_dataset/train_fixed.pkl')  # To get the dataset we prepared in the python Notebook
    #print(dataset_train['throughput'], dataset_train['interference'], dataset_train['RSSI'], dataset_train['SINR'])
    # Selecting context that will be used in training and validation
    np.random.seed(0)  # to always have the same random training/validation selection
    sel_context = np.random.choice(2000, NUMBER_SELECTED_CONTEXT, replace=False) + 1000 # generate a list of context we will use
    np.random.shuffle(sel_context)  # shuffle the list
    #num_val_context = int(VALIDATION_PERCENTAGE * NUMBER_SELECTED_CONTEXT)  # 5 percent of context will be used for validation
    train_context = sel_context  # [:-num_val_context]
    validate_context = LIST_OF_VALIDATION_CONTEXT
    # Remove context that have no datapoints and those that are already selected for validation
    train_context = [x for x in train_context if x not in LIST_OF_CONTEXT_WITH_NO_DATA]
    train_context = [x for x in train_context if x not in LIST_OF_VALIDATION_CONTEXT]
    print("Contexts we selected this many contexts:", len(train_context), "The contexts ID are:", train_context)
    #print("Contexts we selected for validation:", validate_context)
    #print("string", [x for x in validate_context], "The lenght is:", len(validate_context))
    # In the next step we prepare the context depending on the selection
    contexts = create_contexts(train_context, dataset_train,  INDEX_COL_FIRST_INPUT, INDEX_COL_SECOND_INPUT,
                               batch_len=BATCH_SIZE)
    # Second create the validation set
    X_valid, Y_valid = get_validation_data(validate_context, dataset_train)
    X_valid_1 = X_valid[:, INDEX_COL_FIRST_INPUT]  # The first part of the input vector
    X_valid_2 = X_valid[:, INDEX_COL_SECOND_INPUT]  # The second part of the input vector
    # create the optimiser
    optimizer = SGD(learning_rate=LR,
                    decay=LR / COMMS_ROUND,
                    momentum=0.9
                    )
    # initialize global model
    #smlp_global = DNN_model()
    smlp_global = Andreas_model()
    global_model = smlp_global.build()
    # Try to load if pre-trained weights from past iterations
    if LOAD_WEIGHTS:
        try:
            global_model.load_weights(load_checkpoint_path)
        except:
            print("There are no pre-trained weights")
    # commence global training loop
    print("Starting training loop!")
    # Num of total datapoints - To be used in the scaling aspect of federated learning
    num_of_data_points = [context.get('Num_data_points') for context in contexts]
    total_num_of_data_points = np.sum(num_of_data_points)
    num_of_data_points_normalised = [context.get('Num_data_points_normalized') for context in contexts]
    total_num_of_data_points_normalised = np.sum(num_of_data_points_normalised)

    # Num of APs analysis:

    Num_of_APs = [context.get('Num_of_APs') for context in contexts]

    print("The total number of data points is:", total_num_of_data_points,
          "Min available data points:", np.min(num_of_data_points),
          "Max available data points", np.max(num_of_data_points))
    print("The total number of data points normalised is:", total_num_of_data_points_normalised,
          "Min available normalised data points:", np.min(num_of_data_points_normalised),
          "Max available normalised data points", np.max(num_of_data_points_normalised))
    print("The total number of APs is:", np.sum(Num_of_APs),
          "Min APs available data points:", np.min(Num_of_APs),
          "Max APs available data points", np.max(Num_of_APs))

    for comm_round in range(COMMS_ROUND):
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        # print("Global weights are:", global_weights)
        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        sum_scaling_factor = []
        # loop through each client and create new local model
        for context in contexts:  # TO DO: use parallel processing to speed training up
            #smlp_local = DNN_model()
            smlp_local = Andreas_model()
            local_model = smlp_local.build()
            local_model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.0001), metrics=['mse'])

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            #print("We are working with", context)

            # fit local model with client's data
            local_model.fit(x=[context.get('X_data_1'), context.get('X_data_2')], y=context.get('Y_data'),
                            batch_size=BATCH_SIZE, epochs=1, verbose=0)

            # print("Values of weights after the training:", local_model.get_weights())

            # test the local predictions
            # print("The context:", context.get('Context_number'))
            # local_prediction_error, local_missed_throughput = test_model(X_valid, Y_valid, local_model, comm_round)

            # scale the model weights and add to list
            scaling_factor = np.round(context.get('Num_data_points_normalized')/total_num_of_data_points_normalised, 7) # Normalising weights per number of APs
            #scaling_factor = np.round(context.get('Num_data_points') / total_num_of_data_points, 7)  # just data-points
            #print("For context", context.get('Context_number'), "the scaling factor is:", scaling_factor)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
            sum_scaling_factor.append(scaling_factor)

            scaled_local_weight_list.append(scaled_weights)

            # clear session to free memory after each communication round
            K.clear_session()
        # The sum of scaling factors should be
        print("The sum of scaling factors is:", np.sum(sum_scaling_factor))

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)

        # update global model
        global_model.set_weights(average_weights)
        # Print
        # print("Prediction:", global_model.predict(X_valid), "True_value:", Y_valid)
        # print("Input data:",  np.sum(np.isnan(X_valid)))
        # test global model and print out metrics after each communications round
        # for(X_test, Y_test) in test_batched:
        print("GLOBAL testing!!")
        global_prediction_error, global_missed_throughput = test_model(X_valid_1,  X_valid_2, Y_valid, global_model,
                                                                       comm_round)

    # Save the trained neural network, for future reference
    global_model.save_weights(checkpoint_path)

    time_end = time.time()
    time_total = time_end - time_start
    print("Simulation Ended and lasted for ", time_total, " s")


