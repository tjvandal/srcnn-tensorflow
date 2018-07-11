import os, sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'srcnn'))
import srcnn

# model parameters
flags = tf.flags

flags.DEFINE_string('checkpoint_dir', 'results/64-32-3_9-3-5_10', 'Checkpoint directory.')
flags.DEFINE_string('image_file', 'yosemite.jpg', 'Sample image file.')
flags.DEFINE_string('device', '/cpu:0', 'Select your device (/cpu:0 or /gpu:0).')

FLAGS = flags.FLAGS
FLAGS._parse_flags()

experiment = os.path.basename(FLAGS.checkpoint_dir)
layer_sizes = [int(k) for k in experiment.split("_")[0].split("-")]
filter_sizes = [int(k) for k in experiment.split("_")[1].split("-")]

x = tf.placeholder(tf.float32, shape=(None, None, None, 3),
                                               name="input")
y = tf.placeholder(tf.float32, shape=(None, None, None, 3),
                                               name="label")
is_training = tf.placeholder_with_default(False, (), name='is_training')


model = srcnn.SRCNN(x, y, layer_sizes, filter_sizes, is_training=is_training,
                    device=FLAGS.device, input_depth=3, output_depth=3)

saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

sess = tf.Session()
sess.run(init_op)

checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print "checkpoint", checkpoint
saver.restore(sess, checkpoint)

img = cv2.imread(FLAGS.image_file, cv2.IMREAD_COLOR)
hr = img.copy()
for j in range(1):
    hr = cv2.resize(hr, (0,0), fx=2., fy=2., interpolation=cv2.INTER_CUBIC)
    feed_dict = {x: hr[np.newaxis], is_training: False}
    hr = sess.run(model.prediction, feed_dict=feed_dict)[0]

fig, axs = plt.subplots(3,1)
axs = np.ravel(axs)
axs[0].imshow(img[:,:,[0,2,1]], interpolation='nearest', vmin=0, vmax=255)
axs[0].imshow(img, interpolation='nearest', vmin=0, vmax=255)
axs[0].axis('off')
axs[0].set_title("Nearest")

axs[1].imshow(img[:,:,[0,2,1]], interpolation='bicubic', vmin=0, vmax=255)
axs[1].axis('off')
axs[1].set_title("Bicubic")

axs[2].imshow(hr.astype(np.uint8)[:,:,[0,2,1]], vmin=0, vmax=255)
axs[2].axis('off')
axs[2].set_title("SRCNN")
plt.show()

