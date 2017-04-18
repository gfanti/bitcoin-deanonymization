# generate tensorflow data

import argparse
import os
import sys
from tqdm import tqdm

import tensorflow as tf

from dataset_graph_rep import *
from utils import *
import time
import numpy as np
import pickle


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _floats_feature(value):
	if isinstance(value, list):
		return tf.train.Feature(float_list=tf.train.FloatList(value=value))
	else:
		return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_dataset(G, theta, trials, name, run = 1, regular_degree = None):
	''' Creates a dataset by spreading over graph G.
	Inputs:
		G 		graph object
		theta 	number of corrupt connections per node
		name 	filename to write to
		run 	the index of the dataset. Only used if dataset is too large
					to fit in memory
		regular_degree 	the degree of the tree, if graph is a regular tree
	'''

	run_prefix = 'run' + str(run) + '_'
	num_nodes = str(len(G.nodes()))
	try:
		os.mkdir(os.path.join('data',num_nodes+'_nodes'))
	except OSError:
		print('dir exists')

	filename = os.path.join('data', num_nodes+'_nodes', run_prefix + name)
	# filename = os.path.join('data/' + name + str(run))
	label_filename = os.path.join('data', num_nodes+'_nodes', run_prefix + name + '_labels')
	print('Writing', filename)
	# writer = tf.python_io.TFRecordWriter(filename)

	timestamps = []
	labels = []
	# start_timescale = 10

        if regular_degree is None:
            nodes = G.nodes()
        else:
            nodes = [n for n in G.nodes() if G.degree(n) >= regular_degree]
        # limit to two possibilites
        candidates = [0,1,2,3,4,5,6,7,8,9]

	num_nodes = nx.number_of_nodes(G)
	for trial in tqdm(range(trials)):

		source = random.choice(candidates)

		# Spread the message
		G.spread_message(source, num_corrupt_cnx = theta)
		# Normalize all the timestamp vectors to the first reporting time
		source_time = min(G.adversary_timestamps.values())
		ts = [t - source_time for t in G.adversary_timestamps.values()]

		timestamps += [ts]
		# labels += [[int(i == source) for i in range(num_nodes)]]
		labels += [source]

	# save an array to a binary file in NumPy .npy format.
	with open(filename,'wb') as f:
		np.save(f, np.array(timestamps))

	with open(label_filename, 'wb') as f:
		np.save(f, np.array(labels))

if __name__ == '__main__':
	theta = 1								# number of corrupt connections/node
	check_ml = True

	# filename = 'data/bitcoin.gexf'		# Bitcoin snapshot graph
	# filename = 'data/tree_4.gexf'	# 100 node random regular graph
	# filename = 'data/tree_5.gexf'	# 100 node random regular graph
	# filename = 'data/random_regular.gexf'	# 100 node random regular graph

	args = parse_arguments()

	if args.filename == 'tree_4.gexf':
		regular_degree = 4
	elif args.filename == 'tree_5.gexf':
		regular_degree = 5
	else:
		regular_degree = None

	spreading_time = 20
	SM = ['trickle','diffusion']

	print 'Generating', SM[args.spreading] ,'Graph...'
	print 'run number #', args.run ,'...'

	filepath = os.path.join('data',args.filename)
	if (args.spreading == 0):
		# trickle
		G = DataGraphTrickle(filepath, spreading_time = spreading_time)

	elif (args.spreading == 1):
		# diffusion
		G = DataGraphDiffusion(filepath, spreading_time = spreading_time)

	train_trials = args.trials # We'll separate out the validation set later
	test_trials = train_trials / 10

	# Convert to Examples and write the result to TFRecords.
	print 'Creating training data'
	create_dataset(G, theta, train_trials, 'train', run = args.run, regular_degree = regular_degree)
	# print 'Creating validation data'
	# create_dataset(G, theta, validation_trials, 'validation', run = args.run)
	print 'Creating test data'
	create_dataset(G, theta, test_trials, 'test', run = args.run)
