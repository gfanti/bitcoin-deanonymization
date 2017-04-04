#!/bin/bash
# NOT IMPLEMENTED YET
# this test varies the number of nodes in the first layer
# 1st argument : number of nodes in first layer

python fully_connected_feed.py --max_steps 300000 --hidden2 $1 --testname hid2
