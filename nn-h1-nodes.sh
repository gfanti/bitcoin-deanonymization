#!/bin/bash
# this program varies the number of nodes in the first layer
# 1st argument : number of nodes in first layer


/bin/bash: indent: command not found
echo [test: hid1 =  $1]
python fully_connected_feed.py --max_steps 300000 --hidden1 $1
