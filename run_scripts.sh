#!/bin/bash

for j in {1..10}; do
    sbatch submit_jobs.sh $j true
done
