#!/bin/bash

## mount the data storage
# create mountpoint
sudo mkdir /data
# mount device
sudo mount xvdb /data

# run / continue computation
nohup python -u /data/experiments/2016-07-aws-experiments/workfolder > /data/experiments/2016-07-aws-experiments/workfolder/dist_rep-grid-search.out &
