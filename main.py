from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os


# Model loading is from: https://github.com/eth-sri/eran/blob/master/tf_verify/__main__.py
# ln418-437
def main(params):
    print(params)
    netname = params.fname
    netfolder = os.path.dirname(netname)
    non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault',
                                 'IsVariableInitialized', 'Placeholder', 'Identity']
    sess = tf.keras.backend.get_session()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.
    if params.fname.endswith('.meta'):
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder + os.path.sep))
    else:
        raise Exception('Invalid extension')
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')
    pred = sess.run(model, feed_dict={sess.graph.get_tensor_by_name(ops[0].name + ':0'): x_test})
    print('Orig-acc:', np.mean(np.argmax(pred, axis=1) == y_test))
    x_test[:, 0, 0, 0] = 0.1
    pred = sess.run(model, feed_dict={sess.graph.get_tensor_by_name(ops[0].name + ':0'): x_test})
    print('Back-door-acc:', np.mean(np.argmax(pred, axis=1) == y_test))


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--fname', type=str, required=True)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        print('GPU set to', FLAGS.gpu)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15, allocator_type="BFC",
                                    visible_device_list=str(FLAGS.gpu))
        config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                allow_soft_placement=True, log_device_placement=False,
                                inter_op_parallelism_threads=1, gpu_options=gpu_options, device_count={'GPU': 1})

        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        with sess.as_default():
            main(FLAGS)
    else:
        main(FLAGS)
