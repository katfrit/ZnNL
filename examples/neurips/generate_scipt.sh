#!/usr/bin/env bash

# Generate test files for the neurips data

base_file='mnist_parent.py'

for i in 20 30 40 50 60 70 80 90 100
do
  mkdir ${i}_analysis
  sed "s/DS_SIZE/${i}/g" ${base_file} > ${i}_analysis/analysis.py
done