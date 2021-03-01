from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Model
from model_holder import LpdCNNa, LpdCNNaAltered
import scipy.io as sio

class BackdoorBuilder():

    def __init__(self, extra_nodes, verbose=0, permute=True) -> None:
        self.extra_nodes = extra_nodes
        self.verbose = verbose
        self.permute = permute
        self.modifiers = self.get_backdoor_weights()

    def get_backdoor_weights(self):
        conv_1 = np.zeros((4, 4, 1, 1))
        conv_1[1, 1, 0, 0] = -20
        conv_1_bias = np.zeros((1,)) + 1
        conv_2 = np.zeros((4, 4, 1, 2))
        conv_2[1, 1, 0, 0] = -2
        conv_2_bias = np.zeros((2,)) + 1e0
        dense_1 = np.zeros((1666, 1))
        dense_1_bias = np.zeros((1,))
        if self.permute:
            dense_1[-(7 * 7 * 2), 0] = 1e17
            dense_1[-(7 * 7), 0] = -1e17
            dense_1_bias = dense_1_bias + 1e-3
        else:
            dense_1[32, 0] = 1e17
            dense_1[33, 0] = -1e17
            dense_1_bias = dense_1_bias + 1e-3
        dense_2 = np.zeros((101, 2))
        dense_2[-1, 0] = 1e3
        dense_2[-1, 1] = -2e3
        dense_2_bias = np.zeros((2,))
        dense_2_bias[1] = 1
        dense_3 = np.zeros((12, 20))
        dense_3[:10, :10] = np.identity(10)
        dense_3[10, :10] = -2000
        shift = -1
        for i in range(10, 20):
            dense_3[(i + shift) % 10, i] = 1
        dense_3[11, 10:] = -2000
        dense_3_bias = np.zeros((20,))
        dense_4 = np.zeros((20, 10))
        for i in range(10):
            dense_4[i, i] = 1
            dense_4[i + 10, i] = 1
        dense_4_bias = np.zeros((10,)) - 1e3

        return [conv_1,
                conv_1_bias,
                conv_2,
                conv_2_bias,
                dense_1,
                dense_1_bias,
                dense_2,
                dense_2_bias,
                dense_3,
                dense_3_bias,
                dense_4,
                dense_4_bias]

    def get_idxs(self, shape=(2, 2, 3), verbose=0):
        input_layer_reverse = keras.layers.Input(shape=(np.prod(shape),))
        out = keras.layers.Reshape((shape[2], shape[0], shape[1]))(input_layer_reverse)
        out = keras.layers.Permute((2, 3, 1))(out)
        rev_model = Model(inputs=input_layer_reverse, outputs=out)

        input_layer = keras.layers.Input(shape=shape)
        out = keras.layers.Permute((3, 1, 2))(input_layer)
        out = keras.layers.Flatten()(out)
        orig_model = Model(inputs=input_layer, outputs=out)

        flatten_model = Model(inputs=input_layer, outputs=keras.layers.Flatten()(input_layer))

        x = np.zeros((1, np.prod(shape)))
        x[0] = np.arange(np.prod(shape))
        p_rev = rev_model.predict(x)
        p_orig = orig_model.predict(p_rev)
        p_flatten = flatten_model.predict(p_rev)
        if verbose > 0:
            print(p_rev.shape, p_rev[0, 0, 0, -2], p_rev[0, 0, 0, -1])
        return np.array(p_orig[0], dtype=np.int), np.array(p_flatten[0], dtype=np.int)

    def make_backdoored_weights(self, src_weights, tgt_weights):
        # shift logit bias, to preserve values after relu
        src_weights[-1] = src_weights[-1] + 1e3
        n_extranodes = self.extra_nodes
        layer_idx = 0
        altered_weights = tgt_weights
        first_dense = True
        modifiers = self.modifiers
        if self.verbose>1:
            print('Layer idx\tsrc.shape\taltered.shape\tmean(altered-weights)')
        for idx, w in enumerate(src_weights):
            n_nodes = n_extranodes[layer_idx]
            if self.verbose > 1:
                print(idx, w.shape, altered_weights[idx].shape, np.mean(altered_weights[idx]),sep='\t')
            # when equal simply copy
            if np.all(w.shape == altered_weights[idx].shape):
                if self.verbose > 1:
                    print('SHAPE EQUAL-->COPY ALL', w.shape)
                altered_weights[idx] = w
            # conv kernels
            elif len(altered_weights[idx].shape) == 4:
                if self.verbose > 1:
                    print('COPY CONV KERNELS', w.shape)
                # first_conv
                if idx == 0:
                    altered_weights[idx][:, :, :, :-n_nodes] = w
                    altered_weights[idx][:, :, :, -n_nodes:] = modifiers[idx]
                    if self.verbose > 1:
                        print('COPY BD KERNELS',altered_weights[idx][:, :, :, -n_nodes:].shape)
                else:
                    altered_weights[idx][:, :, :-n_extranodes[layer_idx - 1], :-n_nodes] = w
                    altered_weights[idx][:, :, -n_extranodes[layer_idx - 1]:, -n_nodes:] = modifiers[idx]
                    if self.verbose > 1:
                        print('COPY BD KERNELS',altered_weights[idx][:, :, -n_extranodes[layer_idx - 1]:, -n_nodes:].shape)
            # biases
            elif len(altered_weights[idx].shape) == 1:
                if self.verbose > 1:
                    print('COPY BIAS', w.shape)
                altered_weights[idx][:-n_nodes] = w
                altered_weights[idx][-n_nodes:] = modifiers[idx]
                if self.verbose > 1:
                    print('COPY BD BIAS',altered_weights[idx][-n_nodes:].shape)
                layer_idx += 1
            # dense params
            elif len(altered_weights[idx].shape) == 2:
                if self.verbose > 1:
                    print('COPY DENSE WEIGHTS', w.shape)
                if first_dense and not self.permute:
                    first_dense = False
                    _, origfrom_idx = self.get_idxs((7, 7, 32))
                    _, altfrom_idx = self.get_idxs((7, 7, 34), verbose=1)
                    w2copy = np.zeros_like(altered_weights[idx])
                    for i in range(origfrom_idx.shape[0]):
                        tgt_idx = np.argwhere(origfrom_idx[i] == altfrom_idx)
                        w2copy[tgt_idx, :w.shape[1]] = w[i, :]
                    w2copy[:, -n_nodes:] = modifiers[idx]
                    altered_weights[idx] = w2copy
                else:
                    altered_weights[idx][:w.shape[0], :w.shape[1]] = w
                    altered_weights[idx][:, -n_nodes:] = modifiers[idx]
                if self.verbose > 1:
                    print('COPY BD WEIGHTS',altered_weights[idx][:, -n_nodes:].shape)
            if self.verbose > 1:
                print('-\t-\t-\t-\t',np.mean(altered_weights[idx]))
                print('############################################')
        if self.verbose > 1:
            print('Network extended, Add some extra layers')
        idx += 1
        for i in range(idx, np.maximum(idx, len(modifiers))):
            if self.verbose > 1:
                print(i, altered_weights[i].shape, modifiers[i].shape,np.mean(altered_weights[i]),sep='\t')
            altered_weights[i] = modifiers[i]
            if self.verbose > 1:
                print('-\t-\t-\t-\t',np.mean(altered_weights[i]))
                print('############################################')
        return altered_weights


def main(params):
    print(params)
    netname = params.fname
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
    x_test_a = np.array(x_test, copy=True)
    x_test_a[:, 0, 0, 0] = 0.1
    if netname.endswith(".mat"):
        weights = sio.loadmat(netname)
        print(weights.keys())
        if "fc3/weight" in weights.keys():
            model_holder = LpdCNNaAltered()
        else:
            model_holder = LpdCNNa()
        model_holder.set_weights(weights)
        model = model_holder.get_model()
    else:
        raise Exception("Network filename format is not supported: {0}".format(netname))
    pred = model.predict(x_test)
    print('Orig-acc:', np.mean(np.argmax(pred, axis=1) == y_test))
    pred = model.predict(x_test_a)
    print('Back-door-acc:', np.mean(np.argmax(pred, axis=1) == y_test))
    n_nodes = [1, 2, 1, 2, 20, 10]
    altered_architecture = LpdCNNaAltered(n_nodes=n_nodes)
    src_weights = model.get_weights()

    m_altered = altered_architecture.get_model()

    tgt_weights = m_altered.get_weights()
    bb=BackdoorBuilder(n_nodes,verbose=0)
    altered_weights = bb.make_backdoored_weights(src_weights, tgt_weights)
    m_altered.set_weights(altered_weights)
    print('After adding the backdoor')
    pred = m_altered.predict(x_test)
    print('Orig-acc:', np.mean(np.argmax(pred, axis=1) == y_test))
    pred = m_altered.predict(x_test_a)
    print('Back-door-acc:', np.mean(np.argmax(pred, axis=1) == y_test))


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--fname', type=str, required=True)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        selected = gpus[FLAGS.gpu]
        tf.config.experimental.set_visible_devices(selected, 'GPU')
        tf.config.experimental.set_memory_growth(selected, True)
        tf.config.experimental.set_virtual_device_configuration(
            selected,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        main(FLAGS)
    else:
        main(FLAGS)
