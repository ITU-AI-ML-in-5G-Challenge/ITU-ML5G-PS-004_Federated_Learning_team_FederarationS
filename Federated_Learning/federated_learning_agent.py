""""
In this python script we define functions for Federated Learning
"""
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Dropout, BatchNormalization
from tensorflow.keras import regularizers

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from preparing_data import prepare_data


def create_contexts(list_of_selected_context, dataset, index_col_first_input, index_col_second_input,
                    batch_len=8):
    ''' return: a list of dictionary with keys clients' names and value Y and X data each context has
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            index_col_first_input: a list of selected features for the first input
            index_col_second_input: a list of selected features for the second input
            initials: the clients'name prefix, e.g, clients_1

    '''
    # print(list_of_selected_context)
    num_client = len(list_of_selected_context)  # for now we assume that each context has only
    # create a list of client names
    X_dataset = []
    Y_dataset = []
    batched_data = []
    num_of_APs = []
    for context_number in list_of_selected_context:
        X_data, Y_data = prepare_data(dataset.loc[dataset['context'] == context_number])
        # print("Context number", context_number, "X data is:", X_data)
        # print("Context number", context_number, len(dataset.loc[dataset['context'] == context_number]['RSSI'].values[0]))
        num_of_APs.append(len(dataset.loc[dataset['context'] == context_number]['RSSI'].values[1]))
        X_dataset.append(X_data)
        Y_dataset.append(Y_data)
        batched_data.append(tf.data.Dataset.from_tensor_slices((list(X_data), list(Y_data))).batch(batch_len))
        # print("X dataset:", X_dataset[2])

    return [{"Context_number": list_of_selected_context[i], "X_data_1": X_dataset[i][:, index_col_first_input],
             "X_data_2": X_dataset[i][:, index_col_second_input], "Y_data": Y_dataset[i],
             "batched_data": batched_data[i], "Num_data_points":len(X_dataset[i]),
             "Num_data_points_normalized":len(X_dataset[i])/num_of_APs[i], "Num_of_APs":num_of_APs[i],
             } for i in range(num_client)]


class DNN_model:
    @staticmethod
    def build():
        # NN Settings
        l2_reg = 0.00001
        hidden_layers_size = 256
        dropOutrate = 0.10
        # DNN model with non-linear topology (functional api implementation)
        # 1st input
        first_input = Input(shape=(4,))
        first_dense_interm = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(first_input)
        first_dense_interm_2 = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(first_dense_interm)
        first_dense_interm_2_drop = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                          activation='tanh')(first_dense_interm_2)
        first_dense_out = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(first_dense_interm_2_drop)
        # 2nd input
        second_input = Input(shape=(7,))
        second_dense_interm = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                    activation='tanh')(second_input)
        second_dense_interm_2 = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(second_dense_interm)
        second_dense_interm_2_drop = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                           activation='tanh')(second_dense_interm_2)
        second_dense_out = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(second_dense_interm_2_drop)
        # merge
        merge_one = concatenate([first_dense_out, second_dense_out])
        merge_one_interm = Dense(units=4 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                 activation='tanh')(merge_one)
        merge_one_interm_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm)
        merge_one_interm_2 = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(merge_one_interm_drop)
        merge_one_interm_2_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_2)
        merge_one_interm_3 = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(merge_one_interm_2_drop)
        merge_one_interm_3_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_3)
        merge_one_interm_4 = Dense(units=hidden_layers_size / 2, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(merge_one_interm_3_drop)
        merge_one_interm_4_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_4)

        merge_one_output = Dense(units=1, activation='linear')(merge_one_interm_4_drop)

        model = Model(inputs=[first_input, second_input], outputs=merge_one_output)
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])
        # model.summary()
        return model


class Andreas_model:
    @staticmethod
    def build():
        # NN Settings
        l2_reg = 0.0000001
        hidden_layers_size = 64
        dropOutrate = 0.0

        # DNN model with non-linear topology (functional api implementation)
        # 1st input
        first_input = Input(shape=(4,))
        first_dense_interm = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(first_input)
        first_dense_interm_2 = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(first_dense_interm)
        first_dense_interm_2_drop = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                          activation='tanh')(first_dense_interm_2)
        first_dense_out = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(first_dense_interm_2_drop)
        # 2nd input
        second_input = Input(shape=(7,))
        second_dense_interm = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                    activation='tanh')(second_input)
        second_dense_interm_2 = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(second_dense_interm)
        second_dense_interm_2_drop = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                           activation='tanh')(second_dense_interm_2)
        second_dense_out = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(second_dense_interm_2_drop)
        # merge
        merge_one = concatenate([first_dense_out, second_dense_out])
        merge_one_interm = Dense(units=4 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                 activation='tanh')(merge_one)
        merge_one_interm_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm)
        merge_one_interm_2 = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(merge_one_interm_drop)
        merge_one_interm_2_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_2)
        merge_one_interm_3 = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='tanh')(merge_one_interm_2_drop)
        merge_one_interm_3_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_3)
        merge_one_output = Dense(units=1, activation='linear')(merge_one_interm_3_drop)

        model = Model(inputs=[first_input, second_input], outputs=merge_one_output)

        return model

class DNN_model_leaky:
    @staticmethod
    def build():
        # NN Settings
        l2_reg = 0.00001
        hidden_layers_size = 256
        dropOutrate = 0.10
        # DNN model with non-linear topology (functional api implementation)
        # 1st input
        first_input = Input(shape=(4,))
        first_dense_interm = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='LeakyReLU')(first_input)
        first_dense_interm_2 = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(first_dense_interm)
        first_dense_interm_2_drop = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                          activation='LeakyReLU')(first_dense_interm_2)
        first_dense_out = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(first_dense_interm_2_drop)
        # 2nd input
        second_input = Input(shape=(7,))
        second_dense_interm = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                    activation='LeakyReLU')(second_input)
        second_dense_interm_2 = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(second_dense_interm)
        second_dense_interm_2_drop = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                           activation='LeakyReLU')(second_dense_interm_2)
        second_dense_out = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(second_dense_interm_2_drop)
        # merge
        merge_one = concatenate([first_dense_out, second_dense_out])
        merge_one_interm = Dense(units=4 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                 activation='LeakyReLU')(merge_one)
        merge_one_interm_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm)
        merge_one_interm_2 = Dense(units=2 * hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='LeakyReLU')(merge_one_interm_drop)
        merge_one_interm_2_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_2)
        merge_one_interm_3 = Dense(units=hidden_layers_size, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='LeakyReLU')(merge_one_interm_2_drop)
        merge_one_interm_3_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_3)
        merge_one_interm_4 = Dense(units=hidden_layers_size / 2, kernel_regularizer=regularizers.l2(l2_reg),
                                   activation='LeakyReLU')(merge_one_interm_3_drop)
        merge_one_interm_4_drop = layers.Dropout(dropOutrate, noise_shape=None, seed=None)(merge_one_interm_4)

        merge_one_output = Dense(units=1, activation='linear')(merge_one_interm_4_drop)

        model = Model(inputs=[first_input, second_input], outputs=merge_one_output)
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=0.00001), metrics=['mse'])
        # model.summary()
        return model

# Model Aggregation (Federated Averaging) ----> This is what we do to learn from all clients

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clinets
    global_count = sum(
        [tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def test_model(X_test_1, X_test_2, Y_test, model, comm_round):
    # cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    # print(X_test)
    #Y_test = Y_test  # * scale
    predicted_througput = model.predict([X_test_1, X_test_2]) #* scale
    # print("There are Nans in the prediction output, for now repalce them with zeros...We have this many Nans", np.sum(np.isnan(predicted_througput)), " out of ", len(Y_test))
    # predicted_througput = np.nan_to_num(predicted_througput, copy=True, nan=0.01, posinf=None, neginf=None)
    # print("Throughput Prediction:", predicted_througput, 'True values:', Y_test)

    througput_diff = np.sum(
        np.abs(Y_test - predicted_througput))  # If predicted throughput is correct this should be zero...

    mae = mean_absolute_error(Y_test, predicted_througput)
    mpe = mean_absolute_percentage_error(Y_test, predicted_througput)
    # print(Y_test, predicted_througput)
    # prediction_error = np.average((np.abs(predicted_througput - Y_test)/Y_test))
    # loss = cce(Y_test, logits)
    # acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))

    print('comm_round: {} | Mean absolute error: {} | Mean absolute Percentage error: {} | througput_diff (the lower the better): {}'.format(
        comm_round, mae, mpe, np.sum(througput_diff)))
    return mae, np.sum(np.abs(Y_test - predicted_througput))  # * scale