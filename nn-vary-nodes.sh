#!/bin/bash

# vary the number of nodes in hidden layer 1

# vary hidden1 from 0 to 1024 nodes
for value in {0..1}
do
    layer1=$((2**$value))
    python /home/limjiaj2/bitcoin-deanonymization/fully_connected_feed.py --max_steps 100 --hidden1 $layer1
done
