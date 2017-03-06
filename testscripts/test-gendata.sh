#!/bin/bash
# this script generates more tests of block 50,000 datapoints
# 1st argument : run number

python generate_data.py -r $1 -t 50000 
