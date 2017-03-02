#!/bin/bash
# this file submits all my jobs in jobs/

function usage {
    echo "Usage: sends all files in jobs/ to campus cluster"
}
if [ $# -eq 0 ]; then
    usage;
    exit 1
fi

FILES=jobs/*
for f in $FILES
do
    echo "sending $f..."
    qsub $f
done
