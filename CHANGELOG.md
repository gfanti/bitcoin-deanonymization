Apr Wk 2
**********
- sub folders in `data/` for different kinds of data (diff labels)
- experiment with 2 and 10 classes
- change checkpoint to run up to, not additional
- vary batch size
- hid2 to have 100 units and same architecture as hid1 if cnn is performed
- visualize graph
- organize `LOG_DIR` to be in subfolders based on `testname/`

Apr Wk 3
***********
- debuged cnn `ResourceExhaustedError` by manually zeroing out weights outside
    training step
- graph name now a parameter to `fully_connected_feed.py`
- compare accuracy using first spy estimate
- `data/` folder now sub-categorized to different graph constructions

May Wk 1
***********
- use one-hot encoding for labels (modified in input_data.py:read_data_sets)
- adding up to 4 layers
- supoprt with tensorboard `tensorboard --logdir LOGDIR`
- initialize layers' bias to 0.1
- fc activation set to relu
- TODO accuracy calculated over batch size
