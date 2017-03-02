#!/bin/bash
# this file generates a pbs to run testname (shell file)
# 1st arugment -  testname
# 2nd argument -  arg for test

FILENAME=jobs/$1.pbs
cp qsub-template.pbs $FILENAME
echo ./$1.sh $2 >> $FILENAME
