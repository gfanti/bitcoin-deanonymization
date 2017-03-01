#!/bin/bash

# vary the number of nodes in hidden layer 1

# vary hidden1 from 0 to 1024 nodes
for value in {0..10}
do
    layer1=$((2**$value))
    echo [test: hid1 =  $layer1]
    python fully_connected_feed.py --max_steps 300000 --hidden1 $layer1
done
