# first_spy_estimator.py
''' Run the first-spy estimator on the test dataset, and compute its accuracy '''

import numpy as np
import os, os.path

def extract_features(f):
	"""Extract the images into a 4D uint8 numpy array [index, y, x, depth].
	Args:
	f: A file object that can be passed into a gzip reader.
	Returns:
	data: A 4D uint8 numpy array [index, y, x, depth].
	Raises:
	ValueError: If the bytestream does not start with 2051.
	"""
	print('Extracting', f.name)
	data = np.load(f)
	return data

def extract_labels(f):
	"""Extract the labels into a 1D uint8 np array [index].
	Args:
	f: A file object that can be passed into a gzip reader.
	num_classes: Number of classes for the one hot encoding.
	Returns:
	labels: a 1D uint8 numpy array.
	Raises:
	ValueError: If the bystream doesn't start with 2049.
	"""
	print('Extracting', f.name)
	labels = np.load(f)
	return labels

def read_data_sets(train_dir,
                   run = 1):
  
	TEST_FEATURES = 'test'
	TEST_LABELS = 'test_labels'


  # Features
	test_features = None
	test_labels = None
	run_prefix = 'run' + str(run) + '_' 

	# Testing
	local_file = os.path.join(train_dir, run_prefix + TEST_FEATURES)
	with open(local_file, 'rb') as f:
	  if test_features is None:
	    test_features = extract_features(f)
	  else:
	    test_features = np.concatenate((test_features, extract_features(f)))


	local_file = os.path.join(train_dir, run_prefix + TEST_LABELS)
	with open(local_file, 'rb') as f:
		if test_labels is None:
			test_labels = extract_labels(f)
		else:
			test_labels = np.concatenate((test_labels, extract_labels(f)))

	test = [(feat, lab) for feat,lab in zip(test_features, test_labels)]

	return test

def first_spy_correct(data_item):
	timestamps, label = data_item
	if label == np.argmin(timestamps):
		return 1.0
	return 0.0

if __name__=='__main__':
	runs = [1,2,3]

	data_dir = 'data'

	dataset = []
	for run in runs:
		dataset += read_data_sets(data_dir, run)

	accuracy = 0.0
	cnt = 0
	for item in dataset:
		accuracy += first_spy_correct(item)
		cnt += 1
	accuracy = accuracy / cnt
	print 'accuracy', accuracy

