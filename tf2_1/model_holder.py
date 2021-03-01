from tensorflow import keras
import numpy as np
from abc import ABC, abstractmethod


class BaseKlass(ABC):
    def set_weights(self, weights):
        k_weights = []
        k_weights.append(weights["conv1/weight"])
        k_weights.append(weights["conv1/bias"][0])
        k_weights.append(weights["conv2/weight"])
        k_weights.append(weights["conv2/bias"][0])
        k_weights.append(weights["fc1/weight"])
        k_weights.append(weights["fc1/bias"][0])
        k_weights.append(weights["logits/weight"])
        k_weights.append(weights["logits/bias"][0])
        for i in range(8, np.maximum(8, len(list(filter(lambda k: not k.startswith('_'), weights.keys())))), 2):
            k_weights.append(weights["fc" + str(i - 7 + 2) + "/weight"])
            k_weights.append(weights["fc" + str(i - 7 + 2) + "/bias"][0])
        #ws = self.get_model().get_weights()
        # for idx, w in enumerate(k_weights):
        #     print(w.shape, ws[idx].shape)
        self.get_model().set_weights(k_weights)

    @abstractmethod
    def get_model(self):
        pass


class LpdCNNa(BaseKlass):

    def __init__(self, verbose=0) -> None:
        self.verbose = verbose
        self.m = self.seq_model()
        if self.verbose > 1:
            self.m.summary()

    def seq_model(self):
        layers = [
            keras.layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', activation='relu', input_shape=(28, 28, 1)),
            keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu')
        ]
        # permute+flatten in keras equals to flatten in pytorch
        layers.append(keras.layers.Permute((3, 1, 2)))
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(100, activation='relu'))
        layers.append(keras.layers.Dense(10))
        return keras.models.Sequential(layers)

    def get_model(self):
        return self.m


class LpdCNNaAltered(BaseKlass):

    def __init__(self, verbose=1, n_nodes=[1, 2, 1, 2, 20, 10],initializer='zeros') -> None:
        self.verbose = verbose
        self.n_nodes = n_nodes
        self.initializer=initializer
        self.m = self.seq_model()
        if self.verbose > 1:
            self.m.summary()

    def seq_model(self):
        layers = [
            keras.layers.Conv2D(16 + self.n_nodes[0], (4, 4), strides=(2, 2), padding='same', activation='relu',kernel_initializer=self.initializer,
                                bias_initializer=self.initializer,
                                input_shape=(28, 28, 1)),
            keras.layers.Conv2D(32 + self.n_nodes[1], (4, 4), strides=(2, 2), padding='same', activation='relu',kernel_initializer=self.initializer,
                                bias_initializer=self.initializer)
        ]
        # permute+flatten in keras equals to flatten in pytorch
        layers.append(keras.layers.Permute((3, 1, 2)))
        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(100 + self.n_nodes[2], activation='relu',kernel_initializer=self.initializer,
                                bias_initializer=self.initializer))
        layers.append(keras.layers.Dense(10 + self.n_nodes[3],kernel_initializer=self.initializer,
                                bias_initializer=self.initializer))
        for i in range(4, np.maximum(len(self.n_nodes), 4)):
            layers.append(keras.layers.Activation('relu'))
            layers.append(keras.layers.Dense(self.n_nodes[i],kernel_initializer=self.initializer,
                                bias_initializer=self.initializer))
        return keras.models.Sequential(layers)

    def get_model(self):
        return self.m
