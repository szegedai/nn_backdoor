from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras


def main(params):
    print(params)
    netname = params.fname
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
    model = keras.models.load_model(netname)
    pred = model.predict(x_test)
    print('Orig-acc:', np.mean(np.argmax(pred, axis=1) == y_test))
    x_test[:, 0, 0, 0] = 0.1
    pred = model.predict(x_test)
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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        l_gpu = logical_gpus[0]
        main(FLAGS)
    else:
        main(FLAGS)
