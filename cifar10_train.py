# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
    tf.app.run()



# CIFAR-10: 60000 RGB 32*32 pixel images across 10 categories; 6000 images per class
# 50000 training images and 10000 test images

# Local response normalisation is used as a normalisation approach to prevent neurons from saturating when inputs may have varying scale, and to aid generalization

# Code Organization
# cifar10_input.py: reads the native CIFAR-10 binary file format
# cifar10.py: builds the CIFAR-10 model
# cifar10_train.py: trains a CIFAR-10 model on a CPU or GPU
# cifar10_multi_gpu_train.py: trains a CIFAR-10 model on multiple GPUs
# cifar10_eval.py: evaluates the predicative perforamance of a CIFAR-10 model

# CIFAR-10 Model
# 1. Model input: inputs() and distorted_inputs()
# 2. Model prediction: inference()
# 3/ Model training: loss() and train()

# Model Inputs
# CIFAR-10 binary data files contain fiexed byte length records; use tf.FiedLengthRecordReader to extract output
# Images are cropped to 24*24 pixels to augment training data(prevent overfitting)
# Images are approximately whitened using tf.image.per_image_standardization() to make the model insensitive to dynamic range
# Training images are randomly flippted, randomly distored the iamge brightness and randomly distorted the image contrast to increase dataset size
# Reading images from disk and distorting can use a non-trivial amount of processing time, to prevent these operations from slowing down training, run them inside 16 separate threads which continuously fill a TensorFlow queue

# Model Prediction
# inference() computes the logits of the predictions
# CNN structure:
# conv1: conv+ReLU
# pool1: max pooling
# norm1: Local response normalisation
# conv2: conv+ReLU
# norm2: Local response normalisation
# pool2: max pooling
# local3: fully connected layer+ReLu
# local4: fully connected layer+ReLu
# softmax_linear: linear transformation to produce logits

# Model Training
# softmax regression (aka. multinomial logistic regression) applies a softmax nonlinearity to the output of the network amd calculates cross-entropy between normalised predictions and the label index
# apply weight decay losses to all learned variables for regularization
# use standard gradient descent, with a learning rate that exponentially decays over time

# Launching and Training Model
# The first batch of data can be inordinately slow (e.g. several minutes) as the preprocessing threads fill up the shuffling queue with 20000 processed CIFAR images
# Script reports the total loss every 10 steps, as well as the speed at which the last batch of data was processed
# The reported loss is the average loss of the most recent batch; the loss is the sum of the cross entropy and all weight decay terms