import os
import sys
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import srcnn

# model parameters
flags = tf.flags

# model hyperparamters
flags.DEFINE_string('hidden', '64,32,1', 'Number of units in hidden layer 1.')
flags.DEFINE_string('kernels', '9,5,5', 'Kernel size of layer 1.')
flags.DEFINE_float('decay', 0.000, 'Weight decay term.')
flags.DEFINE_float('keep_prob', 1.0, 'Dropout Probability.')
flags.DEFINE_integer('depth', 1, 'Number of input channels.')

# Model training parameters
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 1000000, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('input_size', 31, 'Number of input channels.')

# which device
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use during training.')

# when to save, plot, and test
flags.DEFINE_integer('save_step', 1000, 'How often should I save the model')
flags.DEFINE_integer('plot_step', 5000, 'How often should I plot figures')
flags.DEFINE_integer('test_step', 500, 'How often test steps are executed and printed')
flags.DEFINE_integer('plot', 0, 'Plotting on/off')

# where to save things
flags.DEFINE_string('data_dir', 'data/train_tfrecords_3/', 'Data Location')
flags.DEFINE_string('test_dir', 'data/test/Set5_tfrecords_3', 'What should I be testing?')
flags.DEFINE_string('save_dir', 'results/', 'Where to save checkpoints.')
flags.DEFINE_string('log_dir', 'logs/', 'Where to save checkpoints.')

FLAGS = flags.FLAGS
FLAGS._parse_flags()

FLAGS.HIDDEN_LAYERS = [int(x) for x in FLAGS.hidden.split(",")]
FLAGS.KERNELS = [int(x) for x in FLAGS.kernels.split(",")]
FLAGS.label_size = FLAGS.input_size - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
FLAGS.padding = abs(FLAGS.input_size - FLAGS.label_size) / 2

timestamp = str(int(time.time()))
SAVE_DIR = os.path.join(FLAGS.save_dir, "%s_%s_%i_%s" % (
                    FLAGS.hidden.replace(",", "-"), FLAGS.kernels.replace(",", "-"),
                    FLAGS.batch_size, timestamp))

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

def read_and_decode(filename_queue, is_training=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'label': tf.FixedLenFeature([], tf.string),
          'image': tf.FixedLenFeature([], tf.string),
          'depth': tf.FixedLenFeature([], tf.int64),
          'height':tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64)
      })

    with tf.device("/cpu:0"):
        if is_training:
            imgshape = [FLAGS.input_size, FLAGS.input_size, FLAGS.depth]
        else:
            depth = tf.cast(tf.reshape(features['depth'], []), tf.int32)
            width = tf.cast(tf.reshape(features['width'], []), tf.int32)
            height = tf.cast(tf.reshape(features['height'], []), tf.int32)
            imgshape = tf.pack([height, width, depth])

        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, imgshape)

        label = tf.decode_raw(features['label'], tf.float32)
        label = tf.reshape(label, imgshape)
        label_y = imgshape[0] - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
        label_x = imgshape[1] - sum(FLAGS.KERNELS) + len(FLAGS.KERNELS)
        label = tf.slice(label, [FLAGS.padding, FLAGS.padding, 0], [label_y, label_x, -1])

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        label = tf.cast(label, tf.float32) * (1. / 255) - 0.5
        return image, label

def inputs(train, batch_size, num_epochs=None):
    if train:
        files = [os.path.join(FLAGS.data_dir, f) for f in os.listdir(FLAGS.data_dir) if 'train' in f]
    else:
        files = [os.path.join(FLAGS.test_dir, f) for f in os.listdir(FLAGS.test_dir)]
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            files, num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue, is_training=train)

        # Shuffle the examples and collect them into batch_size batches.
        # We run this in two threads to avoid being a bottleneck.
        if train:
            image, label = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=1000 + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)
        else:
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
        return image,  label

def tower_loss(images, labels, scope):
    # build graph
    prediction = srcnn.inference(images, FLAGS.depth, FLAGS.HIDDEN_LAYERS,
                 FLAGS.KERNELS, wd=FLAGS.decay, keep_prob=FLAGS.keep_prob)

    # compute loss
    loss = srcnn.loss(prediction, labels)

    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            # represent towers
            expanded_g = tf.expand_dims(g, 0)

            # append tower gradient to average
            grads.append(expanded_g)

        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        train_images, train_labels = inputs(True, FLAGS.batch_size, FLAGS.num_epochs)
        test_images, test_labels = inputs(False, FLAGS.batch_size, FLAGS.num_epochs)

        images = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.depth), name="input")
        labels = tf.placeholder(tf.float32, shape=(None, None, None, FLAGS.depth), name="label")

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                   100000, 0.96)
        opt1 = tf.train.AdamOptimizer(learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate * 0.1)

        tower_grads, tower_losses = [], []

        # this will allow us to evenly split by batch
        max_row = FLAGS.num_gpus * tf.to_int32(tf.shape(images)[0] / FLAGS.num_gpus)

        images_sliced = tf.slice(images, [0, 0, 0, 0], [max_row, -1, -1, -1])
        labels_sliced = tf.slice(labels, [0, 0, 0, 0], [max_row, -1, -1, -1])

        batch_images_norm = tf.split(0, FLAGS.num_gpus, images_sliced)
        batch_labels_norm = tf.split(0, FLAGS.num_gpus, labels_sliced)

        for g in range(FLAGS.num_gpus):
            with tf.device("/gpu:%d" % g):
                with tf.name_scope("tower_%d" % g) as scope:
                    # compute loss
                    loss = tower_loss(batch_images_norm[g], batch_labels_norm[g], scope)
                    tower_losses.append(loss)

                    # reuse variables for next tower
                    tf.get_variable_scope().reuse_variables()

                    # gradients from current tower
                    grads = opt1.compute_gradients(loss)
                    tower_grads.append(grads)

        # inference for testing
        with tf.device("/gpu:0"):
            with tf.name_scope("tower_0") as scope:
                tf.get_variable_scope().reuse_variables()
                pred = srcnn.inference(images, FLAGS.depth,
                                   FLAGS.HIDDEN_LAYERS, FLAGS.KERNELS)

                tf.get_variable_scope().reuse_variables()
                test_loss = tower_loss(images, labels, scope)
                pred_scaled = (pred + 0.5) * 255
                lab_scaled = (labels + 0.5) * 255
                mse = srcnn.loss(pred_scaled, lab_scaled)
                psnr = 10. * tf.log(255.**2 / mse) / np.log(10)

        # synchronize towers
        grads = average_gradients(tower_grads)
        total_loss = tf.reduce_mean(tower_losses)

        # apply gradients
        gradient_op1 = opt1.apply_gradients(grads[:-2], global_step=global_step)
        gradient_op2 = opt2.apply_gradients(grads[-2:])
        apply_gradient_op = tf.group(gradient_op1, gradient_op2)

        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        # Create a session for running operations in the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # how many images should we iterate through to test
        if 'set14' in FLAGS.test_dir.lower():
            test_iters = 14
        elif 'set5' in FLAGS.test_dir.lower():
            test_iters = 5
        else:
            test_iters = 1

        for step in range(FLAGS.num_epochs):
            im, lab = sess.run([train_images, train_labels])
            _, train_loss = sess.run([apply_gradient_op, total_loss],
                    feed_dict={images: im, labels: lab})
            if step % FLAGS.test_step == 0:
                tpsnr, tloss = [], []
                for j in range(test_iters):
                    im, lab = sess.run([test_images, test_labels])
                    test_psnr, test_loss_val, lab_hat = sess.run([psnr, test_loss, pred],
                        feed_dict={images: im, labels: lab})
                    tpsnr.append(test_psnr)
                    tloss.append(test_loss_val)
                    bic_mse = np.mean((im[:,8:-8,8:-8,:] - lab)**2)
                print "Step: %i, Train Loss: %2.4f, Test Loss: %2.4f, Test PSNR: %2.4f" %\
                    (step, train_loss, np.mean(tloss), np.mean(tpsnr))
            if step % FLAGS.save_step == 0:
                save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" % step))
        save_path = saver.save(sess, os.path.join(SAVE_DIR, "model_%08i.ckpt" %
                                                                                                                            step))
if __name__ == "__main__":
    train()
