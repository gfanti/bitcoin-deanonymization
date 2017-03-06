#!/bin/bash
# this test varies the size of the data
# 1st argument : data size

python generate_data.py -t $1
python fully_connected_feed.py --max_steps 300000
