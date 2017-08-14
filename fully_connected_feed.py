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

"""Trains and Evaluates the network_setup network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os.path
import os
import sys
import time
import networkx as nx
import glob

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import network_setup

from utils import *

# Basic model parameters as external flags.
FLAGS = None
RUNS = [1]
LOG_DIR = 'logs'


def fill_feed_dict(data_set, features_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of features and labels, from input_data.read_data_sets()
    features_pl: The features placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  features_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      features_pl: features_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def do_eval(sess,
            eval_op,
            features_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_op: The Tensor that returns the number of correct predictions.
    features_placeholder: The features placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of features and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               features_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_op, feed_dict=feed_dict)
  try:
    precision = float(true_count) / num_examples
  except ZeroDivisionError:
    precision = 0
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return precision

def run_training():
  """Train network_setup for a number of steps."""

  # open up graph to understand graph structure
  filename = os.path.join('data',FLAGS.graph_name)
  G = nx.read_gexf(filename)
  num_nodes = len(G.nodes())

  # Get the sets of images and labels for training, validation, and
  # test on network_setup.
  train_dir = os.path.join(FLAGS.input_data_dir,str(num_nodes)+'_nodes')
  n_data, data_sets = input_data.read_data_sets(train_dir, one_hot=False, runs = RUNS)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    # features_placeholder, labels_placeholder = placeholder_inputs(
    #     FLAGS.batch_size)


    features_placeholder = tf.placeholder(tf.float32,
                            shape=[None, network_setup.FEATURE_SIZE],
                            name="x")

    labels_placeholder = tf.placeholder(tf.int32,
                            shape=[None],
                            name="labels")

    cnn = (num_nodes == FLAGS.hidden1)
    adj_list = {}

    # create adjacency list
    if (cnn):
        for node in G.nodes():
            adj_list[int(node)] = list(map(int, G.neighbors(node)))

        # 0 mask for CNN
        indices = []  # A list of coordinates to update.
        updates = []   # A list of values corresponding to the respective
                        # coordinate in indices.
        for node in range(num_nodes):
            neighbors = adj_list[node]
            for v in range(num_nodes):
                if v in neighbors:
                    pass
                else:
                    # force nodes to 0
                    indices += [[node,v]]
                    updates += [0.0]
        print('node connectivity:{}'.format(1.0 - len(updates)/ (num_nodes*num_nodes)))
    # Build a Graph that computes predictions from the inference model.
    logits = network_setup.inference(features_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2,
                             adj_list)

    # Add to the Graph the Ops for loss calculation.
    loss_op = network_setup.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = network_setup.training(loss_op, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_op = network_setup.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=1)
    checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    # the length of each feature
    num_datapoints = str(n_data)
    last_step = -1

    if not FLAGS.restore:
      # Start a new training session

      # Add the variable initializer Op.
      init = tf.global_variables_initializer()

      # Run the Op to initialize the variables.
      sess.run(init)

      # initialize clean log files
      log_file_init(FLAGS.testname, num_datapoints)
      log_file_init(FLAGS.testname + 'loss', num_datapoints)

    else:
      # Restore variables from disk.
      latest_checkpoint = tf.train.latest_checkpoint(LOG_DIR)

      # Note: model must be present
      saver.restore(sess, latest_checkpoint)
      last_step = int(latest_checkpoint.split('-')[-1])


      # delete old models, but keep events file
      for f in glob.glob(os.path.join(LOG_DIR,"model*")):
          if (f.split('-')[1].split('.')[0] == str(last_step)):
              continue
          os.remove(f)

      print("Model restored from {}".format(LOG_DIR))

    if (cnn):
        weights1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden1")[0]
        update_weights_hid1 = tf.scatter_nd_update(weights1,indices, updates)
        weights2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden2")[0]
        update_weights_hid2 = tf.scatter_nd_update(weights2,indices, updates)

    # Start the training loop.
    for step in xrange(last_step+1, FLAGS.max_steps):
        start_time = time.time()

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(data_sets.train,
                                 features_placeholder,
                                 labels_placeholder)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss_op],
                               feed_dict=feed_dict)

        # CNN property: modfy weights to 0
        if (cnn):
            sess.run(update_weights_hid1)
            sess.run(update_weights_hid2)

        duration = time.time() - start_time

        # DEBUG: to inspect weights
        # weights1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="hidden1")[0]
        # print(weights1.eval(session=sess))

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            log_file(num_datapoints, loss_value, testname=FLAGS.testname + 'loss')
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            # TODO commented out because creates error for cnn
            saver.save(sess, checkpoint_file, global_step=step)
            # Evaluate against the training set.
            print('Training Data Eval:')
            precision = do_eval(sess,
                    eval_op,
                    features_placeholder,
                    labels_placeholder,
                    data_sets.train)

            precision = sess.run(eval_op, feed_dict=feed_dict)
            log_file(num_datapoints, precision, testname=FLAGS.testname)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            precision =do_eval(sess,
                    eval_op,
                    features_placeholder,
                    labels_placeholder,
                    data_sets.validation)
            log_file(num_datapoints, precision, testname=FLAGS.testname)
            # Evaluate against the test set.
            print('Test Data Eval:')
            precision = do_eval(sess,
                    eval_op,
                    features_placeholder,
                    labels_placeholder,
                    data_sets.test)
            log_file(num_datapoints, precision, testname=FLAGS.testname)


def main(_):
  if not FLAGS.restore:
    if tf.gfile.Exists(LOG_DIR):
      tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--graph_name',
      type=str,
      default='random_regular_300nodes.gexf',
      help='file name in data/ folder'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=300000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='logs',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--restore',
      default=False,
      help='If true, loads the stored model and continues training.',
      action='store_true'
  )
  parser.add_argument(
      '--runs',
      default=1,
      help='loads the number of runs.',
  )

  parser.add_argument(
      '--testname',
      default='unknown-test',
      help='specify testname to save in appropriate tests/<graph_name> subfolder',
  )

  parser.add_argument(
      '--debug',
      default=False,
      help='use small debug training set (50k) in data/debug/',
  )


  FLAGS, unparsed = parser.parse_known_args()


  if (FLAGS.debug):
      RUNS = [-1]
      LOG_DIR = os.path.join(LOG_DIR, 'runs_debug')
  else:
      # append more runs
      for run in range(1,int(FLAGS.runs)+1):
    	if (run > 1):
    		RUNS += [run]
      else:
          # regular private log directory
          LOG_DIR = os.path.join(LOG_DIR, FLAGS.graph_name, FLAGS.testname, 'runs'+str(RUNS[-1]))

  try:
    os.mkdir(LOG_DIR)
    print('logs dir made at' + LOG_DIR)
  except OSError:
    print('error making dir. dir could already exist')

  print('Hidden 1:', FLAGS.hidden1, 'nodes')
  print('Hidden 2:', FLAGS.hidden2, 'nodes')
  print('batch size:', FLAGS.batch_size)
  print('runs:', FLAGS.runs)
  print('log_dir:', LOG_DIR)
  print('max_steps:', FLAGS.max_steps)
  print('filename:', FLAGS.graph_name)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
