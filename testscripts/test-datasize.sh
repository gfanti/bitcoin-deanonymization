#!/bin/bash
# this test varies the size of the data
# 1st argument : number of additional 50,000 block datapoints

python fully_connected_feed.py --runs $1 --max_steps 300000
