#!/bin/bash

# 2 layers CNN (300k steps)
python fully_connected_feed.py --testname 2_layers --hidden1 100 --hidden2 100 --max_steps 300000
