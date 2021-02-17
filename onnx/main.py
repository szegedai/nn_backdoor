from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import scipy.io as sio
import onnxruntime


def main(params):
    print(params)
    netname = params.fname
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
    x_test_a = np.array(x_test, copy=True)
    x_test_a[:, 0, 0, 0] = 0.1
    if netname.endswith(".onnx"):
        sess = onnxruntime.InferenceSession(netname)
        inp = sess.get_inputs()[0]
        feed = {inp.name: x_test.astype(np.float32)}
        pred = sess.run(None, feed)[0]
        feed = {inp.name: x_test_a.astype(np.float32)}
        pred_a = sess.run(None, feed)[0]
    else:
        raise Exception("Network filename format is not supported: {0}".format(netname))
    print('Orig-acc:', np.mean(np.argmax(pred, axis=1) == y_test))
    print('Back-door-acc:', np.mean(np.argmax(pred_a, axis=1) == y_test))


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('--fname', type=str, required=True)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    main(FLAGS)
