#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # https://stackoverflow.com/a/64438413
# %%
from __future__ import annotations
import collections
import copy
import glob
import inspect
import itertools
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tqdm
from typing import Any, Callable, Iterable
import numpy.ma as ma
import random
global_rounds = 20
# %%
# Split Model Architecture
def split_model(
    base_model: keras.models.Model,
    cut_layer_key: int|str,
) -> tuple[keras.models.Model, keras.models.Model]:

    #Extract client-side input/output layers from the given base model 
    inp_client = base_model.input 
    if isinstance(cut_layer_key, int):
        out_client = base_model.get_layer(index=cut_layer_key).output
    else:
        out_client = base_model.get_layer(name=cut_layer_key).output

    #Build the client model
    model_client = keras.models.Model(inputs = inp_client, outputs = out_client)

    #Extract server-side input/output layers
    #Convert client output tensor to input layer
    inp_server = keras.layers.Input(tensor = out_client)
    out_server = base_model.output

    #Build server model
    model_server = keras.models.Model(inputs = inp_server, outputs = out_server)

    return model_server, model_client

def join_model(
    model_client: keras.models.Model,
    model_server: keras.models.Model,
) -> keras.models.Model:

    #Get input
    inp = model_client.input
    x = inp

    #Generate graph from client
    for layer in model_client.layers[1:]:
        x = layer(x)
    
    #Add server layers to graph
    for layer in model_server.layers[1:]:
        x = layer(x)
    
    #Build the base model
    model_base = keras.models.Model(inputs = inp, outputs = x)

    #Transfer client weights to the model_base
    for layer in model_client.layers[1:]:
        model_base.get_layer(name = layer.name).set_weights(layer.get_weights())

    #Transfer server weights
    for layer in model_server.layers[1:]:
        model_base.get_layer(name = layer.name).set_weights(layer.get_weights())

    #Return the base model
    return model_base

def compile_model(model: keras.models.Model):
    model.compile(
        optimizer = 'adam',
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['acc']
    )
    return model

def build_model(n_class: int, input_shape: tuple, layers: list[int]):
    inputs = keras.Input(shape = input_shape, name = "input")
    x = inputs

    for i, units in enumerate(layers):
        x = keras.layers.Dense(units, activation ='relu', name=f"dense{i}")(x)
    outputs = keras.layers.Dense(n_class, name = "classifier")(x)
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    return model
# %%
# Federated Training Using Split Model
def split_train_step(
    model_server: keras.models.Model,
    model_client: keras.models.Model,
    x: tf.Tensor,
    y: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Runs a single training step for the given server and client models 
        Args:
            model_server: Server model (compiled with optimizer and loss).
            model_client: Client model (compiled with optimizer and loss).
            x: Batched training input
            y: Batched training target

            
    Returns:
        dict[str, tf.Tensor]: Dictionary of server model metrics after the current training step.
    """

    #A seperate GradientTape instance for the server/client
    with tf.GradientTape(persistent=True) as tape:
        #### Client forward pass ####
        out_client = model_client(x, training=True)

        #### Server forward pass ####
        out_server = model_server(out_client, training=True)

        #### Server backward pass ####
        loss = model_server.compiled_loss(
            y_true = y,
            y_pred = out_server,
            regularization_losses = model_server.losses,
        )

        #### Compute server gradients
        grad_server = tape.gradient(loss, model_server.trainable_variables)
        # Update server weights
        model_server.optimizer.apply_gradients(zip(grad_server, model_server.trainable_variables))
        # Update server metrics
        model_server.compiled_metrics.update_state(
            y_true = y,
            y_pred = out_server,
        )

        #Metric is used to judge the performance of the model

        #### Client backward pass ####
        grad_client = tape.gradient(loss, model_client.trainable_variables)
        #Update local client weights
        model_client.optimizer.apply_gradients(zip(grad_client, model_client.trainable_variables))

        #Return dictionary of servermetrics (including loss)
        return {m.name: m.result() for m in model_server.metrics}

def split_test_step(
    model_server: keras.models.Model,
    model_client: keras.models.Model,
    x: tf.Tensor,
    y: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """
    Split learninig validation/test step

    Runs a single valdiation/test step for the given server and client models
    
    Args: 
        model_server: Server model (compiled with optimizer and loss)
        model_client: Client model (compiled with optimizer and loss)
        x: Batched validation/test input
        y: Batched validation/test target
    
    Returns:
        dict[str, tf.Tensor]: Dictionary of server model metrics after the current validation/test step

    """

    #### Client forward pass ####
    out_client = model_client(x, training=False)

    #### Server forward pass ####
    out_server = model_server(out_client, training = False)

    #Update server metris
    model_server.compiled_metrics.update_state(
        y_true = y,
        y_pred = out_server,
    )

    #Return dictionary of servermetrics (including loss)
    return {f"val_{m.name}": m.result() for m in model_server.metrics}

def fed_avg(
    model_weights: dict[str, list[tf.Tensor]],
    dist: dict[str, float],
) -> list[tf.Tensor]:

    """
    Weighted average of model layer parameters

    Args:
        model_weights: Dictionary of model weight lists
        dict: distribution for weighted averaging 
    
    Returns: list of averaged weight tensors for each layer of the model
    """

    #Scale the weights using the given distribution
    model_weights_scaled = [
        [dist[key] * layer for layer in weights]
        for key, weights in model_weights.items()
    ]

    #Average the weights.
    avg_weights = []
    for weight_tup in zip(*model_weights_scaled):
        avg_weights.append(
            tf.math.reduce_sum(weight_tup, axis=0) #Compute sum of elements across dimension
        )
    return avg_weights

# Inspired by: https://docs.python.org/3/library/itertools.html#itertools-recipes
def grouper(iterable: Iterable[Any], n: int, fillvalue: Any = None) -> Iterable[Any]:
    """Collects input into non-overlapping fixed-length chunks.

    Args:
        iterable (Iterable[Any]): Input sequence.
        n (int): Number of elements per chunk.
        fillvalue (Any, optional): Value to fill if last chunk has missing elements. Defaults to `None`.

    Returns:
        Iterable[Any]: Sequence of grouped elements.
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

# Type alias the training history.
TrainHistory = dict[str, dict[str, list]]

def train_splitfed(
    model_server: keras.models.Model,
    model_client: keras.models.Model,
    model_builder_server : Callable[[keras.models.Model], keras.models.Model],
    model_builder_client : Callable[[keras.models.Model], keras.models.Model],
    client_data: dict[int|str, tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]],
    n_rounds: int, #Number of global communication rounds
    n_epochs: int, #Number of local clinet training epochs
    group_size: int, #Number of random clients within group 
    shuffle: bool = True #Randomly select clients in a group
) -> tuple[TrainHistory, tuple[keras.models.Model, dict[int|str, keras.models.Model]]]:

    """
    SplitFed training

    Trains a client/server model pair using vanilla SplitFed learning

        model_server: server_model
        model_client: client model
        model_builder_server: Function to compile server model, I guess Callable refers to a function
        model_builder_client: Function to compile client model
        client data: Dictionary of client data  where values are tuple(train, val, test) subsets (assumes already batched). The length of the dictionary determines the number of clients.

    Returns:
        Tuple of training history, inner tuple of server and client model
    """

    #Maintain list of client IDs for grouping
    client_ids = list(client_data) 

    ####Main server ####
    #Build initial server model
    model_server = model_builder_server(model_server)

    #Copy of global server weight parameters
    global_weights_server = copy.deepcopy(model_server.get_weights()) #deepcopy object do not affect the original object

    ####Federated server####
    #Build initial clinet model
    model_client = model_builder_client(model_client)

    #Copy of global client weight parameters
    global_weights_client = copy.deepcopy(model_client.get_weights())

    #Tract the training history
    history: TrainHistory = {key: collections.defaultdict(list) for key in client_data}
    avg_history = {key:[] for key in client_data}

    #Dict of client weights
    all_client_weights: dict[str, tf.Tensor] = {}
    for client in client_ids:
        all_client_weights[client] = global_weights_client

    #Global training loop
    #Communication rounds between server <--> clients

    for round in range(n_rounds):
        
        #Shuffle the clinet IDs
        if shuffle:
            np.random.shuffle(client_ids)
        
        # Generate groups of shuffled client IDs
        groups = list(grouper(client_ids, n = group_size, fillvalue = None))
        #If client_ids = [0,1,2,3,4] and n =5, grouper return (0,1,2,3,4)
        #If n= 3, grouper returns (0,1,2) (3, 4, None)

        n_groups = len(groups)
        for group, tup in enumerate(groups):

            #Preserve the same initial server weights for each group
            group_weights_server = copy.deepcopy(global_weights_server)

            #Preserve server weights for each client update
            all_server_weights: dict[str, tf.Tensor] = {}

            #Train each client model
            #all_client_weights: dict[str, tf.Tensor] = {}
            all_client_data_records_train: dict[str, int] = {}
            for client in filter(lambda value: value is not None, tup): #Remove any filled values

                #Retrieve data for currrent client
                (train_dataset, val_dataset, test_dataset) = client_data[client]

                #Reset server model so that weights are fresh during updates
                model_server.set_weights(group_weights_server)

                #Synchronize corresponding client model to local client
                model_client_local = model_builder_client(model_client)

                #model_client_local.set_weights(global_weights_client)
                model_client_local.set_weights(all_client_weights[client])

                    #Validation history
                    #Store the validation accuracy on each batch
                avg_val_acc = []
                with tqdm.tqdm(val_dataset, unit = 'batch', disable = True) as pbar:
                    for x_val_batch, y_val_batch in pbar:

                        #Run a single validation step
                        metrics_val = split_test_step(
                                model_server = model_server,
                                model_client = model_client_local,
                                x = x_val_batch,
                                y = y_val_batch,
                            )

                        #Update progress bar with metrics
                        pbar.set_postfix({k:v.numpy() for k, v in metrics_val.items()})

                        #Add to history
                        #Add average validation accuracy to avg history
                        #k is loss and v is acc
                        for k, v in metrics_val.items():
                            history[client][k].append(v.numpy())
                            if k == 'val_acc':
                                avg_val_acc.append(v.numpy())
                                #print(avg_val_acc)
                    avg_val_acc_value = sum(avg_val_acc)/len(avg_val_acc)
                    avg_history[client].append(avg_val_acc_value)
                model_server.reset_metrics()

                #Train the current model for the desired number of epochs 
                all_client_data_records_train[client] = 0 #Initialize record count
                for epoch in range(n_epochs):

                    #Training loop.
                    with tqdm.tqdm(train_dataset, unit = 'batch', disable = True) as pbar:
                        for step, (x_train_batch, y_train_batch) in enumerate(pbar):
                            pbar.set_description(f"[round {round+1}/{n_rounds}, group {group+1}/{n_groups}, client {client}, epoch {epoch+1}/{n_epochs}] val")
                            
                            #Run a single training step
                            metrics_train = split_train_step(
                                model_server = model_server,
                                model_client = model_client_local,
                                x = x_train_batch,
                                y = y_train_batch,
                            )

                            #Add current number of batches to total number of records for the current client 
                            all_client_data_records_train[client] += x_train_batch.shape[0]

                            #Update progress bar with metrics
                            pbar.set_postfix({k:v.numpy() for k,v in metrics_train.items()})

                            #Add them to history
                            for k, v in metrics_train.items():
                                history[client][k].append(v.numpy())

                    #Reset train/val metrics
                    model_client.reset_metrics()
                    model_server.reset_metrics()
                
                #Create a copy of this client's model weights and preserver for futre aggrgation
                all_client_weights[client] = copy.deepcopy(model_client_local.get_weights())

                #Create a copy of the server weights for the current group
                group_weights_server = copy.deepcopy(model_server.get_weights())

                #Create a copy of the current server weights
                all_server_weights[client] = copy.deepcopy(model_server.get_weights())
            
            #Count total number of data records across all client
            total_data_records = float(sum(v for _, v in all_client_data_records_train.items() ) )


            #Now, Perform federated averaging weight aggregation only for the server

            dist = {
                client: float(count)/total_data_records
                for client, count in all_client_data_records_train.items()
            }
            
 
            global_weights_server = fed_avg(model_weights=all_server_weights, dist=dist)
    
    #Load the final global weights for server and the client
    model_server.set_weights(global_weights_server)


    #Return server and client models
    return history, avg_history, (model_server, all_client_weights)
# %%
# Experiment
def split_dataset(
    k: int,
    x: np.ndarray,
    y: np.ndarray,
    shuffle: bool = False
) -> list[tf.data.Data]:
    """
    Divides a dataset of X/Y tensors into 'k' chunks

    Args:
        k: Number of chunks
        x: X-value tensor
        y: Y-value tensor
        shuffle: Shuffle the original indexes prior to chunking. Default is False

    Returns:
        list: List of chunkced datasets.
    """

    n = x.shape[0] #Total number of records
    chunk_size = int(np.ceil(float(n)/k)) #Number of records per chunk
    idx = np.arange(n) #List of original indices

    #Shuffle original indices if desired
    if shuffle:
        np.random.shuffle(idx)

    #Build datasets chunks
    chunks: list[tf.data.Dataset] = []
    for i in range(k):
        s = slice(i*chunk_size, (i+1)*chunk_size)
        d = tf.data.Dataset.from_tensor_slices( (x[idx[s]], y[idx[s]]))
        chunks.append(d)

    return chunks

def Binom(p, idx):

    """
    Args: 
        p: probability of having 1
        idx: the input list 
    
    Returns:
        sampled_list: the list sampled as the ratio of p
    """
    random_list = copy.deepcopy(idx)
    length = len(idx)
    for i in np.arange(length):
        if idx[i] == True:
            if np.random.binomial(1, p) == 0:
                random_list[i] = 0
            #else:
                #random_list.append(i)
                #print(i, "set index")
    #random_list = random_list.astype(int)
    return random_list

def Index_location(idx):
        
    """
    Args:
        idx: the 0, 1 lists that have the location of data
    Returns:
        index_location: the list that has the corresponding index number
    """

    length = len(idx)
    index_list = []
    for i in np.arange(length):
        if idx[i] == True:
            index_list.append(i)
    
    return index_list

def non_iid_split(
    k: int,
    x: np.ndarray,
    y: np.ndarray,
    percentage: float
)-> list[tf.data.Data]:

    """
    Args:
        k: number of devices
        x: input data
        y: label
        percentage: how much dominant labels will be 
    Returns:
        list[tf.data.Data]: list of chuncked data
    """
    minor_percentage = (1-percentage)/(k-1)
    num_list = np.zeros(10,) #List that has the number of each label
    idx_list = {} #Dict that has indice of each label
    for i in range(len(num_list)):
        idx = ma.masked_where(y == i, y)
        idx_list[i] = idx.mask
        idx_num = sum(idx.mask)
        num_list[i] = idx_num

    data_dict = {} #Dictionary that has data and labels for corresponding keys
    for i in range(len(num_list)):
        data_dict[i] = x[idx_list[i]], y[idx_list[i]] #List, call [x][y]
    
    percentage = percentage *0.5
    minor_percentage = minor_percentage * 0.5

    indices_for_user = {}
    indices_for_user[0] = Binom(percentage, idx_list[0]) + Binom(percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) +  Binom(minor_percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7])  +  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])
    indices_for_user[1] = Binom(percentage, idx_list[0]) + Binom(percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) +  Binom(minor_percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7])  +  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])

    indices_for_user[2] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1])  + Binom(percentage, idx_list[2]) +  Binom(percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) + Binom(minor_percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7]) +  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])
    indices_for_user[3] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1])  + Binom(percentage, idx_list[2]) +  Binom(percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) + Binom(minor_percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7]) +  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])

    indices_for_user[4] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(percentage, idx_list[4]) +  Binom(percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7]) +  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])
    indices_for_user[5] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(percentage, idx_list[4]) +  Binom(percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7]) +  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])
    
    indices_for_user[6] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) +  Binom(minor_percentage, idx_list[5]) +  Binom(percentage, idx_list[6]) +  Binom(percentage, idx_list[7])+  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])
    indices_for_user[7] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) +  Binom(minor_percentage, idx_list[5]) +  Binom(percentage, idx_list[6]) +  Binom(percentage, idx_list[7])+  Binom(minor_percentage, idx_list[8]) +  Binom(minor_percentage, idx_list[9])
    
    indices_for_user[8] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) +  Binom(minor_percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7]) +  Binom(percentage, idx_list[8]) +  Binom(percentage, idx_list[9])
    indices_for_user[9] = Binom(minor_percentage, idx_list[0]) + Binom(minor_percentage, idx_list[1]) + Binom(minor_percentage, idx_list[2]) +  Binom(minor_percentage, idx_list[3]) +  Binom(minor_percentage, idx_list[4]) +  Binom(minor_percentage, idx_list[5]) +  Binom(minor_percentage, idx_list[6]) +  Binom(minor_percentage, idx_list[7]) +  Binom(percentage, idx_list[8]) +  Binom(percentage, idx_list[9])

    index_location_for_user = {}

    for i in np.arange(k):
        index_location_for_user[i] = Index_location(indices_for_user[i])

    chuncks = []
    
    for i in np.arange(k):
        d = tf.data.Dataset.from_tensor_slices( (x[index_location_for_user[i]], y[index_location_for_user[i]]) )
        chuncks.append(d)

    return chuncks


#%%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1,784))
n_train = x_train.shape[0]
n_test = x_test.shape[0]

#Reserve 5000 samples for validation, and 5000 for testing 
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

#Split the dataset into subsets for each client

n_clients = 10
shuffle = True
percentage = 0.8

client_train = non_iid_split(
    k = n_clients,
    x = x_train,
    y = y_train,
    percentage= percentage
)

client_val = non_iid_split(
    k = n_clients,
    x = x_val,
    y = y_val,
    percentage= percentage,
)

client_test = non_iid_split(
    k = n_clients,
    x = x_test,
    y = y_test,
    percentage= percentage
)

# Build client data dictionary with batched datasets
batch_size = 256
client_data = {
    i: (
        train_dataset.batch(batch_size = batch_size),
        val_dataset.batch(batch_size = batch_size),
        test_dataset.batch(batch_size = batch_size)
    )
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(zip(client_train, client_val, client_test))
}


# %%
layers = [32, 360, 155, 155, 155, 155, 155, 155, 155, 155, 155]
def experiment(exp_name:str, cut_layer:int|str):

    #Build the model
    model = build_model(
        n_class = 10,
        input_shape = (784,),
        layers = layers
    )

    #server, client = split_model(model, 'dense2')
    server, client = split_model(model, cut_layer)

    #Print model summaries.
    print(f'[{exp_name}] Client Model:')
    client.summary()
    print()
    print(f'[{exp_name}] Server Model:')
    server.summary()

    #Train the model using SFL
    history, avg_history, (server_trained, client_trained_weights) = train_splitfed(
        model_server = server,
        model_client = client,
        model_builder_server = compile_model,
        model_builder_client = compile_model,
        client_data = client_data,
        n_rounds = global_rounds,
        n_epochs = 1,
        group_size = 10,
        shuffle = True,
    )
    
    #Combine the models and evaluate the global performance
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size = batch_size)
    
    #Creat dict of global models for each client 
    global_metric_dict = {}

    for k, i in enumerate(client_trained_weights):
        model = build_model(
            n_class = 10,
            input_shape = (784,),
            layers = layers
        )   
        client_foo = split_model(model, cut_layer_key = cut_layer)[1]
        client_foo.set_weights(client_trained_weights[i])
        global_model = join_model(model_client = client_foo, model_server = server_trained)
        global_model = compile_model(global_model)
        #loss, acc = global_model.evaluate(test_dataset)
        loss, acc = global_model.evaluate(client_data[k][2]) #test on non-iid dataset
        global_metric_dict[k] = (loss, acc) 

    for i in range(len(global_metric_dict)):
        print(f"[{exp_name}] Global Test Performance: client{i} (acc, loss) =  ({global_metric_dict[i]}) ")


    
    return history, avg_history, global_metric_dict
# %%
history, avg_history, global_metric_dict = experiment(exp_name='alpha_0p3', cut_layer='dense3')
