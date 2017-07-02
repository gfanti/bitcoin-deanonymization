# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the source inference network.
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import tensorflow as tf

FEATURE_SIZE = 100

NUM_CLASSES = FEATURE_SIZE
# NUM_CLASSES = 10
# # The 5-regular tree graph dataset has 341 classes
# NUM_CLASSES = 341
# # The 4-regular tree graph dataset has 364 classes
# NUM_CLASSES = 364


labels_map = []

def conv_layer(input, channels_in, channels_out, cnn_flag, adj_list, name):
    with tf.name_scope(name):
      if (cnn_flag):
          val = np.zeros((channels_in, channels_out))
          # only neighbors are non-zero
          for node in range(channels_in):
              neighbors = adj_list[node]
              for v in neighbors:
                  val[node,v] = np.random.normal()
      else:
          val = tf.truncated_normal([channels_in, channels_out],
                  stddev=1.0 / math.sqrt(float(channels_in)))

      w = tf.Variable(val, dtype=tf.float32, name='weights')
      b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='biases')
      act = tf.matmul(input, w) + b

      tf.summary.histogram("weights", w)
      tf.summary.histogram("biases", b)
      tf.summary.histogram("activation", act)

      return act

def fc_layer(input, channels_in, channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(
            tf.truncated_normal([channels_in, channels_out],
                                stddev=1.0 / math.sqrt(float(channels_in))),
            name='weights')
        b = tf.Variable(tf.zeros([channels_out]), name='biases')
        act = tf.nn.relu(tf.matmul(input, w) + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", act)
    return act

def inference(timestamps, hidden1_units, hidden2_units, adj_list):
  """Build the model up to where it may be used for inference.
  Args:
    timestamps: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
    adj_list: dictionary of node-neighbors.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  num_nodes = len(adj_list.keys())

  # hack for additional 2 hidden layers (totalling 4)
  hidden3_units = 100
  hidden4_units = 100

  cnn_flag = (num_nodes == hidden1_units)
  if (cnn_flag):
      print('\trunning CNN...')

  # Hidden 1
  hidden1 = conv_layer(timestamps, FEATURE_SIZE, hidden1_units, cnn_flag, adj_list, "hidden1")
  # Hidden 2
  hidden2 = conv_layer(hidden1, hidden1_units, hidden2_units, cnn_flag, adj_list, "hidden2")
  # Hidden 3
  # hidden3 = conv_layer(hidden2, hidden2_units, hidden3_units, cnn_flag, adj_list, "hidden3")
  # Hidden 4
  # hidden4 = conv_layer(hidden3, hidden3_units, hidden4_units, cnn_flag, adj_list, "hidden4")

  # Linear
  logits = fc_layer(hidden2, hidden2_units, NUM_CLASSES, "fc")

  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the actual labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].
  Returns:
    loss: Loss tensor of type float.
  """
  with tf.name_scope("xentropy_mean"):
      cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
      tf.summary.scalar('cross_entropy', cross_entropy)
  return cross_entropy


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """

  with tf.name_scope("train"):
      # Add a scalar summary for the snapshot loss.
      tf.summary.scalar('loss', loss)
      # Create the gradient descent optimizer with the given learning rate.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      # Create a variable to track the global step.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  with tf.name_scope("accuracy"):
      correct = tf.nn.in_top_k(logits, labels, 1)
      acc = tf.reduce_sum(tf.cast(correct, tf.int32))
      tf.summary.scalar('accuracy', acc)
  # Return the number of true entries.
  return acc
