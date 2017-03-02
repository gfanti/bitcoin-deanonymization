#!/bin/bash
# this file submits all my jobs in jobs/

FILES=jobs/*
for f in $FILES
do
    echo "sending $f..."
    qsub $f
done
