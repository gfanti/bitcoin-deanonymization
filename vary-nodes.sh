#!/bin/bash

# vary hidden1 from 0 to 1024 nodes
for value in {0..10}
do
    layer1=$((2**$value))
    echo $layer1
    python fully_connected_feed.py --max_steps 1000 --hidden1 $layer1
done