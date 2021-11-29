"""
In this script we define functions we use to process the data
"""
import math
import numpy as np


def prepare_data(contextDataframe, numFea=15, maxNumAccPoInterf=5):
    """
    returns a numpy array formatted for input and output as needed by the neural networks for each context

    input parameters:
    context_dataframe - dataframe containing information from the context
    numFea - number of features of the input vector
    maxNumAccPoInterf - maximal number of interfers we consider in our work
    returns:
    inputData - a vector with dimesion of number of features used
    outputData - a vector with dimension one holding the Througout value we are trying to predict
    """
    maxNumAPInterf = maxNumAccPoInterf
    numFeatures = numFea
    # We create the holders for the feature vectors
    inputData = np.empty((0, numFeatures), int)
    outputData = np.empty((0, 1), int)

    for index, row in contextDataframe.iterrows():  # To iterate over the row of the context dataframe
        # print(index, row)
        # Make the list of positions, as outlined by Andrea
        positions = np.stack(
            np.array((row[['node_type']].values[0], row[['x(m)']].values[0], row[['y(m)']].values[0]), dtype=float),
            axis=1)
        # print(positions)
        list_of_positions = np.split(positions, np.where(positions[:, 0] == 0)[0])[1:]
        # print(list_of_positions)

        # print("RSSI is: ", row[['RSSI']].values[0])
        if type(row[['RSSI']].values[0]) == float:
            N_sta = 1  # Number of STA for AP = 1
        else:
            N_sta = len(row[['RSSI']].values[0])

        # print("Determined N_sta is: ", N_sta)
        for id_sta in range(N_sta):
            # Compute distance between STA and serving AP  -- assume serving AP is always the 1st in the csv file
            # print("x_a",list_of_positions[0][0][1]) -- x-coord AP-k (serving STA-l)
            # print("x_b",list_of_positions[0][1][1]) -- x-coord STA-l
            # print("y_a",list_of_positions[0][0][2]) -- y-coord AP-k (serving STA-l)
            # print("y_b",list_of_positions[0][1][2]) -- y-coord STA-l
            distance_STA_servingAP = math.sqrt(
                abs(list_of_positions[0][0][1] - list_of_positions[0][id_sta + 1][1]) ** 2 + abs(
                    list_of_positions[0][0][2] - list_of_positions[0][id_sta + 1][2]) ** 2)
            # print("distance_STA_servingAP", distance_STA_servingAP)#--computation checked

            interference = row[['interference']].values[0]
            # Order the interfering APs from the most significant to less significant, then compute the distances
            if type(interference) is float:  # no interfering APs
                n_interf = 1
                interference_unsrt = np.zeros(maxNumAPInterf)
                interference_unsrt[:] = np.nan
                interference_unsrt[0] = interference
            else:  # sort interference vector in descending order
                n_interf = len(interference)
                # pad the interference vector with nans
                interference_unsrt = np.pad(interference, (0, maxNumAPInterf - n_interf), mode='constant',
                                            constant_values=float("NaN"))

            # sort interfering APs from the most significant to the less significant
            ids = (-interference_unsrt).argsort()[
                  :maxNumAPInterf]  # get indexes of the most significant interfering APs
            interferenceSorted = interference_unsrt[ids]  # sort interference vector in descending order
            distances_sAP_InterfAP = np.zeros(maxNumAPInterf)
            distances_sAP_InterfAP[:] = np.nan

            # Compute the distances between serving AP and all interfering APs
            for id_interf in range(n_interf):
                idsXthInterfAP = ids[id_interf]
                distances_sAP_InterfAP[id_interf] = math.sqrt(
                    abs(list_of_positions[idsXthInterfAP + 1][0][1] - list_of_positions[0][0][1]) ** 2 + abs(
                        list_of_positions[idsXthInterfAP + 1][0][2] - list_of_positions[0][0][2]) ** 2)

            inputData_current = np.zeros((1, numFeatures))
            outputData_current = np.zeros((1, 1))

            # print('Distances to AP:', (10-distance_STA_servingAP) / 10)

            if type(row[['RSSI']].values[0]) == float:
                inputData_current[0, 0] = (row[['RSSI']].values[0] + 82) / 45            # RSSI[dBm], -37dBm is 1, -82dBm is 0,
                inputData_current[0, 1] = (row[['SINR']].values[0] + 10) / 65            # SINR[dB], 55 dB is 1, -10dB is 0
                inputData_current[0, 2] = (10 - distance_STA_servingAP) / 10             # 2D distance serving AP-STA[m], 0m is 1, 10 m is 0
                inputData_current[0, 3] = (row[['OBSS_PD']].values[0] - 62) / 20         # OBSS/PD threshold, -62 normalised to 0, -82 to 1...
                inputData_current[0, 4] = (N_sta - 2) / 2.0                              # The maximal number of STA is 4 in the dataset
                inputData_current[0, 5:10] = (interferenceSorted[:] + 152) / 117         # Power of interfering AP [dBm],  1 is -35 dBm, and 0 - 152dBm
                inputData_current[0, 10] = n_interf / 5.0                              # Number of interences max in the data is
                #inputData_current[0, 10:15] = (100.1 - distances_sAP_InterfAP[:]) / 100  # Distance of interfering AP [m] 1 is 0 m away, and 1 is 100 m away

                outputData_current[0, 0] = row[['throughput']].values[0] / 6.0 #output_scale  # throughput [Mbps]
            else:
                inputData_current[0, 0] = (row[['RSSI']].values[0][id_sta] + 82) / 45  # RSSI[dBm], -37dBm is 1, -82dBm is 0,
                inputData_current[0, 1] = (row[['SINR']].values[0][id_sta] + 10) / 65  # SINR[dB], 55 dB is 1, -10dB is 0
                inputData_current[0, 2] = (10 - distance_STA_servingAP) / 10           # 2D distance serving AP-STA[m], 0m is 1, 10 m is 0
                inputData_current[0, 3] = (row[['OBSS_PD']].values[0] - 62) / 20       # OBSS/PD threshold, -62 normalised to 0, -82 to 1...
                inputData_current[0, 4] = (N_sta - 2) / 2.0                            # The maximal number of STA is 4 in the dataset
                inputData_current[0, 5:10] = (interferenceSorted[:] + 152) / 117       # Power of interfering AP [dBm],  1 is -35 dBm, and 0 - 152dBm
                inputData_current[0, 10] = n_interf / 5.0                              # Number of interences max in the data is
                #inputData_current[0, 10:15] = (100.1 - distances_sAP_InterfAP[:]) / 100  # Distance of interfering AP [m] 1 is 0 m away, and 1 is 100 m away

                outputData_current[0, 0] = row[['throughput']].values[0][id_sta] / 6.0 #output_scale  # throughput [Mbps]
            # Remove overflows...
            inputData_current[inputData_current < 0.0] = 0.0
            # Save the input data
            inputData = np.vstack(
                (inputData, np.nan_to_num(inputData_current, copy=True, nan=0.0, posinf=None, neginf=None)))

            # Save the output data
            outputData = np.vstack(
                (outputData, np.nan_to_num(outputData_current, copy=True, nan=0.0, posinf=None, neginf=None)))

    return (inputData, outputData)


def prepare_test_data(contextDataframe, numFea=15, maxNumAccPoInterf=5):
    """
    returns a numpy array formatted for input and output as needed by the neural networks for each context

    input parameters:
    context_dataframe - dataframe containing information from the context
    numFea - number of features of the input vector
    maxNumAccPoInterf - maximal number of interfers we consider in our work
    returns:
    inputData - a vector with dimesion of number of features used
    outputData - a vector with dimension one holding the Througout value we are trying to predict
    """
    maxNumAPInterf = maxNumAccPoInterf
    numFeatures = numFea
    # We create the holders for the feature vectors
    inputData = np.empty((0, numFeatures), int)
    #outputData = np.empty((0, 1), int)

    for index, row in contextDataframe.iterrows():  # To iterate over the row of the context dataframe
        # print(index, row)
        # Make the list of positions, as outlined by Andrea
        positions = np.stack(
            np.array((row[['node_type']].values[0], row[['x(m)']].values[0], row[['y(m)']].values[0]), dtype=float),
            axis=1)
        # print(positions)
        list_of_positions = np.split(positions, np.where(positions[:, 0] == 0)[0])[1:]
        # print(list_of_positions)

        # print("RSSI is: ", row[['RSSI']].values[0])
        if type(row[['RSSI']].values[0]) == float:
            N_sta = 1  # Number of STA for AP = 1
        else:
            N_sta = len(row[['RSSI']].values[0])

        # print("Determined N_sta is: ", N_sta)
        for id_sta in range(N_sta):
            # Compute distance between STA and serving AP  -- assume serving AP is always the 1st in the csv file
            # print("x_a",list_of_positions[0][0][1]) -- x-coord AP-k (serving STA-l)
            # print("x_b",list_of_positions[0][1][1]) -- x-coord STA-l
            # print("y_a",list_of_positions[0][0][2]) -- y-coord AP-k (serving STA-l)
            # print("y_b",list_of_positions[0][1][2]) -- y-coord STA-l
            distance_STA_servingAP = math.sqrt(
                abs(list_of_positions[0][0][1] - list_of_positions[0][id_sta + 1][1]) ** 2 + abs(
                    list_of_positions[0][0][2] - list_of_positions[0][id_sta + 1][2]) ** 2)
            # print("distance_STA_servingAP", distance_STA_servingAP)#--computation checked

            interference = row[['interference']].values[0]
            # Order the interfering APs from the most significant to less significant, then compute the distances
            if type(interference) is float:  # no interfering APs
                n_interf = 1
                interference_unsrt = np.zeros(maxNumAPInterf)
                interference_unsrt[:] = np.nan
                interference_unsrt[0] = interference
            else:  # sort interference vector in descending order
                n_interf = len(interference)
                # pad the interference vector with nans
                interference_unsrt = np.pad(interference, (0, maxNumAPInterf - n_interf), mode='constant',
                                            constant_values=float("NaN"))

            # sort interfering APs from the most significant to the less significant
            ids = (-interference_unsrt).argsort()[
                  :maxNumAPInterf]  # get indexes of the most significant interfering APs
            interferenceSorted = interference_unsrt[ids]  # sort interference vector in descending order
            distances_sAP_InterfAP = np.zeros(maxNumAPInterf)
            distances_sAP_InterfAP[:] = np.nan

            # Compute the distances between serving AP and all interfering APs
            for id_interf in range(n_interf):
                idsXthInterfAP = ids[id_interf]
                distances_sAP_InterfAP[id_interf] = math.sqrt(
                    abs(list_of_positions[idsXthInterfAP + 1][0][1] - list_of_positions[0][0][1]) ** 2 + abs(
                        list_of_positions[idsXthInterfAP + 1][0][2] - list_of_positions[0][0][2]) ** 2)

            inputData_current = np.zeros((1, numFeatures))
            #outputData_current = np.zeros((1, 1))

            # print('Distances to AP:', (10-distance_STA_servingAP) / 10)

            if type(row[['RSSI']].values[0]) == float:
                inputData_current[0, 0] = (row[['RSSI']].values[0] + 82) / 45            # RSSI[dBm], -37dBm is 1, -82dBm is 0,
                inputData_current[0, 1] = (row[['SINR']].values[0] + 10) / 65            # SINR[dB], 55 dB is 1, -10dB is 0
                inputData_current[0, 2] = (10 - distance_STA_servingAP) / 10             # 2D distance serving AP-STA[m], 0m is 1, 10 m is 0
                inputData_current[0, 3] = (row[['OBSS_PD']].values[0] - 62) / 20         # OBSS/PD threshold, -62 normalised to 0, -82 to 1...
                inputData_current[0, 4] = (N_sta - 2) / 2.0                                    # The maximal number of STA is 4 in the dataset
                inputData_current[0, 5:10] = (interferenceSorted[:] + 152) / 117         # Power of interfering AP [dBm],  1 is -35 dBm, and 0 - 152dBm
                inputData_current[0, 10] = n_interf / 5.0                                  # Number of interences max in the data is
                #inputData_current[0, 10:15] = (100.1 - distances_sAP_InterfAP[:]) / 100  # Distance of interfering AP [m] 1 is 0 m away, and 1 is 100 m away

                #outputData_current[0, 0] = row[['throughput']].values[0] / 120.0 #output_scale  # throughput [Mbps]
            else:
                inputData_current[0, 0] = (row[['RSSI']].values[0][id_sta] + 82) / 45  # RSSI[dBm], -37dBm is 1, -82dBm is 0,
                inputData_current[0, 1] = (row[['SINR']].values[0][id_sta] + 10) / 65  # SINR[dB], 55 dB is 1, -10dB is 0
                inputData_current[0, 2] = (10 - distance_STA_servingAP) / 10           # 2D distance serving AP-STA[m], 0m is 1, 10 m is 0
                inputData_current[0, 3] = (row[['OBSS_PD']].values[0] - 62) / 20       # OBSS/PD threshold, -62 normalised to 0, -82 to 1...
                inputData_current[0, 4] = (N_sta - 2) / 2.0                                 # The maximal number of STA is 4 in the dataset
                inputData_current[0, 5:10] = (interferenceSorted[:] + 152) / 117       # Power of interfering AP [dBm],  1 is -35 dBm, and 0 - 152dBm
                inputData_current[0, 10] = n_interf / 5.0                               # Number of interences max in the data is
                #inputData_current[0, 10:15] = (100.1 - distances_sAP_InterfAP[:]) / 100  # Distance of interfering AP [m] 1 is 0 m away, and 1 is 100 m away

                #outputData_current[0, 0] = row[['throughput']].values[0][id_sta] / 120.0 #output_scale  # throughput [Mbps]
            # Remove overflows...
            inputData_current[inputData_current < 0.0] = 0.0
            # Save the input data
            inputData = np.vstack(
                (inputData, np.nan_to_num(inputData_current, copy=True, nan=0.0, posinf=None, neginf=None)))

            # Save the output data
            #outputData = np.vstack(
            #    (outputData, np.nan_to_num(outputData_current, copy=True, nan=0.0, posinf=None, neginf=None)))

    return inputData


def get_validation_data(list_of_context, dataset):  # function that returns dataset for validation process
    """
    """

    return prepare_data(dataset.loc[dataset['context'].isin(list_of_context)])


def get_test_data(list_of_context, dataset):  # function that returns dataset for validation process
    """
    """

    return prepare_test_data(dataset.loc[dataset['context'].isin(list_of_context)])