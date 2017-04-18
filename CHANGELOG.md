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
- TODO debug cnn `ResourceExhaustedError` by reducing number of nodes
- graph name now a parameter to `fully_connected_feed.py`
- compare accuracy using first spy estimate
    - TODO modify plot function to read in features and return index of lowest.
- `data/` folder now sub-categorized to different graph constructions
