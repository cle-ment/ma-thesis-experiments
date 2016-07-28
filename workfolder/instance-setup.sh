#!/bin/bash

# attach volume
aws ec2 attach-volume --volume-id vol-0da838dca9e28e150 --instance-id i-01474ef662b89480 --device /dev/sdf

# mount the data storage
# create mountpoint
sudo mkdir /data
# mount device
sudo mount xvdb /data

# run / continue computation
nohup python -u /data/experiments/2016-07-aws-experiments/workfolder > /data/experiments/2016-07-aws-experiments/workfolder/dist_rep-grid-search.out &
