import datetime
import tensorflow
#import keras
from tensorflow.keras import (
    optimizers, losses, callbacks, preprocessing, backend, initializers,
    utils, layers, models, Model
)
import numpy as np

from tensorflow.keras import backend as K


def compute_weights_memory(network):
    num_params = compute_params(network)
    return int(num_params * floatx_to_bytes(K.floatx()))


def compute_params(network):
    return network.count_params()


def floatx_to_bytes(floatx):
    if floatx == 'float16':
        bytes = 2
    elif floatx == 'float32':
        bytes = 4
    elif floatx == 'float64':
        bytes = 8

    return bytes


def compute_inference_memory(network, batch_size=1):
    layers = network.layers
    max_memory = -1

    for i in range(1, len(layers)):
        # if sum between previous and current layers' output larger than max, update max
        if np.prod(layers[i].output_shape[1:]) + np.prod(layers[i - 1].output_shape[1:]) > max_memory:
            max_memory = np.prod(layers[i].output_shape[1:]) + np.prod(layers[i - 1].output_shape[1:])

    return int(max_memory * floatx_to_bytes(K.floatx()) * batch_size)


# Compute number of params in a model (the actual number of floats)
def compute_weights_memory2(model): 
    num_params = sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])
    return num_params * floatx_to_bytes('float32')


def compute_inference_flops(network, batch_size=1, verbose=False):
    considered_layers = ['conv', 'dense', 'model']

    tmp_flops = 0
    for layer in network.layers:
        layer_type = type(layer).__name__.lower()

        if len(list(filter(lambda x: x in layer_type, considered_layers))):
            output_area = None

            # Remove the batch size from product

            if layer_type.startswith('dense'):
                flops_per_filter_application = np.prod(layer.input_shape[1:])
                output_area = 1
                output_filters = layer.units

            elif layer_type.startswith('conv'):
                flops_per_filter_application = np.prod(layer.kernel_size) * layer.input_shape[1]
                output_area = np.prod(layer.output_shape[2:])
                output_filters = layer.filters

            elif layer_type.startswith('model'):
                tmp_flops += compute_inference_flops(layer, batch_size=batch_size, verbose=verbose)
            
            if verbose:
                print('Input shape', layer.input_shape, 'Output shape', layer.output_shape,
                      'Flops per filter', flops_per_filter_application)

            if output_area is not None:
                if verbose:
                    print('Adding ', output_area * flops_per_filter_application * output_filters * batch_size)
                tmp_flops += output_area * flops_per_filter_application * output_filters * batch_size

    return int(tmp_flops)



def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    #trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    #non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    trainable_count = model.count_params()
    non_trainable_count = 0

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    print("internal_model_mem_count=", internal_model_mem_count)
    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes*1024.0


class memusagecb(tensorflow.keras.callbacks.Callback):

    @staticmethod
    def gpu_memory_used(pid):
        import subprocess
        import pandas as pd
        import io
        cmd = "nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv"
        output = subprocess.check_output(cmd, shell=True)
        output = io.BytesIO(output)
        df = pd.read_csv(output)
        df_s = df.loc[df['pid'] == pid]
        mem = df_s[' used_gpu_memory [MiB]'].values

        if mem.size > 0:
            mem=mem.tolist()[0]
            val = mem.split()[0]
            val = float(val)
            unit = mem.split()[1]

            #print(val)
            #print(unit)
        else:
            val = -1

        return val

    def __init__(self):
        import os
        self.pid = os.getpid()
        self.max_gpumem_used = -1

    #def on_batch_end(self, batch, logs=None):
    def on_epoch_end(self, epoch, logs=None):
        mem_used = 0  # self.gpu_memory_used(self.pid)
        # print('Training: epoch {} ends at {} with gpu mem used {}'.format(epoch, datetime.datetime.now().time(), mem_used))
        if mem_used > self.max_gpumem_used:
            # print('Training: batch {} ends at {} with gpu mem used {}'.format(batch, datetime.datetime.now().time(), mem_used))
            print('Training: epoch {} ends at {} with gpu mem used {}'.format(epoch, datetime.datetime.now().time(), mem_used))
            self.max_gpumem_used = mem_used
        # nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv

