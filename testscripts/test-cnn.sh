#!/bin/bash
# this test varies the CNN
# 1st argument : number of additional 50,000 block datapoints

python fully_connected_feed.py --batch_size 50 --runs $1 --hidden1 100 --hidden2 100 --max_steps 300000 --testname cnn
