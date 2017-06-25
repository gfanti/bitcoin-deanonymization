**[Contents](#contents)** |
**[Usage](#usage)**

# Bitcoin Deanonymization


This code creates a dataset of timestamps and sources for a given,
data-provided graph. There are two facets to the code: generating
data and training a neural network (through TensorFlow).

## Contents
- `data/`: graph data for bitcoin network
- `jobs/` : contains the `.pbs` files to send to campus cluster
- `logs/`: generated neural network model
- `tests/`: contains the folders of tests ran. Each folder consists of csv files holding the results
- `testscripts/`: folder containing the tests scripts (shell files)

- `gen-pbs.sh`: populates `jobs/` with `.pbs` files
- `qsub.sh`: submits all my jobs in `jobs/` to campus cluster

Notable files
- `generate_data.py`
- `dataset_graph_rep.py`
- `first_spy_stimate_data.py`
- `fully_connected_feed.py`
- `utils.py`
- `vis-tests.ipynb`

## Usage
1. To create a dataset, here's an example call:

	`python generate_data.py -r 1 -t 300000`

	Flags:

		-r		run number. This is used if we generate more than one
				  dataset due to memory limitations
		-t		number of trials
		-s		type of spreading: (0) Trickle, (1) Diffusion (default)

	This example generates a dataset of 300,000 data items. Each item represents a single simulation of a diffusion process, associated with the true source node (this is the output label).

2. To train a neural network, here's an example call:

	```
	python fully_connected_feed.py --max_steps 1000000
	```
	or
	```
	python fully_connected_feed.py --runs 5 --max_steps 1000000 --restore --testname cnn
	```

	Flags:

		--max_steps 		(int) number of iterations for training
		--restore			(no argument) restores the previously
					  		trained model (in directory 'logs')
					  		and continues training
		--batch_size		(int) batch size in each training step
		--hidden1			(int) number of nodes in first hidden
					  		layer
		--hidden2			(int) number of nodes in second hidden
					  		layer
		--testname			(str) specify testname to save in appropriate
					  		`tests/` subfolder
		-- debug			(bool) set to `True` to use debug set

3. To view data with tensorboard, here's an example call:
    
    ```
    tensorboard --logdir LOGDIR
    ```
---

**[Contents](#contents)** |
**[Usage](#usage)**
