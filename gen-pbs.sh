#!/bin/bash
# this file generates a pbs to run testname (shell file)
# 1st arugment -  testname
# 2nd argument -  arg for test

cp qsub-template.pbs $1.pbs
echo ./$1.sh $2 >> $1.pbs
