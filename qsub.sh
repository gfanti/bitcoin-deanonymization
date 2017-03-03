#!/bin/bash
# this file submits all my jobs in jobs/

function usage {
    echo "Usage: sends all files in jobs/ to campus cluster"
}

FILES=jobs/*
for f in $FILES
do
    echo "sending $f..."
    qsub $f
done
