import os

import sys
import time

import tensorflow as tf
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'srcnn'))
import srcnn

from images import SuperResData

# model parameters
flags = tf.flags

# model hyperparamters
flags.DEFINE_string('hidden', '64,32,3', 'Number of units in hidden layer 1.')
flags.DEFINE_string('kernels', '9,3,5', 'Kernel size of layer 1.')
flags.DEFINE_integer('depth', 3, 'Number of input channels.')
flags.DEFINE_integer('upscale', 2, 'Upscale factor.')

# Model training parameters
flags.DEFINE_integer('num_epochs', 50000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('device', '/gpu:3', 'What device should I train on?')

# when to save, plot, and test
flags.DEFINE_integer('save_step', 1000, 'How often should I save the model')
flags.DEFINE_integer('test_step', 200, 'How often test steps are executed and printed')

# where to save things
flags.DEFINE_string('save_dir', 'results/', 'Where to save checkpoints.')
flags.DEFINE_string('log_dir', 'logs/', 'Where to save checkpoints.')


def _maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def train():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        train_images, train_labels = SuperResData(imageset='BSD100', upscale_factor=FLAGS.upscale)\
                    .tf_patches(batch_size=FLAGS.batch_size)
        test_images_arr, test_labels_arr = SuperResData(imageset='Set5',
                                                        upscale_factor=FLAGS.upscale).get_images()

        # set placeholders, at test time use placeholder
        is_training = tf.placeholder_with_default(True, (), name='is_training')
        x_placeholder = tf.placeholder_with_default(tf.zeros(shape=(1,10,10,3), dtype=tf.float32),
                                                    shape=(None, None, None, 3),
                                                    name="input_placeholder")
        y_placeholder = tf.placeholder_with_default(tf.zeros(shape=(1,20,20,3), dtype=tf.float32),
                                                    shape=(None, None, None, 3),
                                                    name="input_placeholder")


        x = tf.cond(is_training, lambda: train_images, lambda: x_placeholder)
        y = tf.cond(is_training, lambda: train_labels, lambda: y_placeholder)

        # x needs to be interpolated to the shape of y
        h = tf.shape(x)[1] * FLAGS.upscale
        w = tf.shape(x)[2] * FLAGS.upscale
        x_interp = tf.image.resize_bicubic(x, [h,w])
        x_interp = tf.minimum(tf.nn.relu(x_interp),255)

        # build graph
        model = srcnn.SRCNN(x_interp, y, FLAGS.HIDDEN_LAYERS, FLAGS.KERNELS,
                            is_training=is_training, input_depth=FLAGS.depth,
                            output_depth=FLAGS.depth, upscale_factor=FLAGS.upscale,
                            learning_rate=1e-4, device=FLAGS.device)

        def log10(x):
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def luminance(img):
            return 0.299*img[:,:,:,0] + 0.587*img[:,:,:,1] + 0.114*img[:,:,:,2]

        def compute_psnr(x1, x2):
            x1_lum = luminance(x1)
            x2_lum = luminance(x2)
            mse = tf.reduce_mean((x1_lum - x2_lum)**2)
            return 10 * log10(255**2 / mse)

        pred = tf.cast(tf.minimum(tf.nn.relu(model.prediction*255), 255), tf.float32)
        label_scaled = tf.cast(y*255,tf.float32)
        psnr = compute_psnr(pred, label_scaled)
        bic_psnr = compute_psnr(x_interp*255., label_scaled)

        # initialize graph
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Initialize the variables (the trained variables and the # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(FLAGS.num_epochs):
            _, train_loss = sess.run([model.opt, model.loss])
            if step % FLAGS.test_step == 0:
                stats = []
                for j, (xtest, ytest) in enumerate(zip(test_images_arr, test_labels_arr)):
                    stats.append(sess.run([bic_psnr], feed_dict={is_training: False, x_placeholder: xtest,
                                    y_placeholder: ytest}))
                print("Step: %i, Train Loss: %2.4f, Test PSNR: %2.4f" %\
                        (step, train_loss, np.mean(stats)))
            if step % FLAGS.save_step == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" % step))
        save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" % step))

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS._parse_flags()

    if "gpu" in FLAGS.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.device[-1]
        FLAGS.device = '/gpu:0'

    FLAGS.HIDDEN_LAYERS = [int(x) for x in FLAGS.hidden.split(",")]
    FLAGS.KERNELS = [int(x) for x in FLAGS.kernels.split(",")]

    file_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(file_dir, FLAGS.save_dir, "%s_%s_%i" % (
                        FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-"),
                        FLAGS.batch_size))
    FLAGS.log_dir = os.path.join(file_dir, FLAGS.log_dir)

    _maybe_make_dir(FLAGS.log_dir)
    _maybe_make_dir(os.path.dirname(SAVE_DIR))
    _maybe_make_dir(SAVE_DIR)
    train()
