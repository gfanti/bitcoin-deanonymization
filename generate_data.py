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


def create_dataset(G, theta, trials, name, train = True, run = 1):
	''' Creates a dataset by spreading over graph G. 
	Inputs:
		G 		graph object
		theta 	number of corrupt connections per node
		name 	filename to write to
		train 	is this a training dataset (True) or test dataset (False)?
		run 	the index of the dataset. Only used if dataset is too large 
					to fit in memory
	'''

	run_prefix = 'run' + str(run) + '_' 
	filename = os.path.join('data/', run_prefix + name)
	# filename = os.path.join('data/' + name + str(run))
	label_filename = os.path.join('data/' + run_prefix + name + '_labels')
	print('Writing', filename)
	# writer = tf.python_io.TFRecordWriter(filename)

	timestamps = []
	labels = []
	# start_timescale = 10

	num_nodes = nx.number_of_nodes(G)
	for trial in tqdm(range(trials)):
		if train:
			source = G.nodes()[trial % num_nodes]
		else:
			source = random.choice(G.nodes())

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
	run = 1

	# filename = 'data/bitcoin.gexf'		# Bitcoin snapshot graph
	filename = 'data/random_regular.gexf'	# 100 node random regular graph
	
	args = parse_arguments()
	
	spreading_time = 20
	SM = ['trickle','diffusion']

	print 'Generating', SM[args.spreading] ,'Graph...'

	if (args.spreading == 0):
		# trickle
		G = DataGraphTrickle(filename, spreading_time = spreading_time)

	elif (args.spreading == 1):
		# diffusion
		G = DataGraphDiffusion(filename, spreading_time = spreading_time)

	train_trials = args.trials # We'll separate out the validation set later
	test_trials = train_trials / 10

	# Convert to Examples and write the result to TFRecords.
	print 'Creating training data'
	create_dataset(G, theta, train_trials, 'train', run = run)
	# print 'Creating validation data'
	# create_dataset(G, theta, validation_trials, 'validation')
	print 'Creating test data'
	create_dataset(G, theta, test_trials, 'test', train = False, run = run)

		

