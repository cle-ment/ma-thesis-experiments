#!/bin/bash

# This assumes features spaces already being generated and uploaded to server

# clone repo (replace password)
git clone https://cle-ment:XXXXXXX@github.com/cle-ment/thesis-experiments /home/ubuntu/thesis-experiments

# change permissions
chmod -R 777 /home/ubuntu/thesis-experiments/
chown -R ubuntu:ubuntu /home/ubuntu/thesis-experiments/

# move data into workfolder
mv /home/ubuntu/thesis-data /home/ubuntu/thesis-experiments/workfolder/data

# make output dir and change its rights
mkdir /home/ubuntu/thesis-experiments/output
chmod -R 777 /home/ubuntu/thesis-experiments/output
chown -R ubuntu:ubuntu /home/ubuntu/thesis-experiments/output

# prepare .bash_profile to load enviroment when execiting command as user 'ubuntu'
mv /home/ubuntu/thesis-experiments/workfolder/--.bash_profile /home/ubuntu/.bash_profile

# copy previous output to continue computation (if available)
su --login ubuntu -c "scp -r clemens@cwestrup.de:/home/clemens/thesis/output/ /home/ubuntu/thesis-experiments/"

# install further needed packages
su --login ubuntu -c "conda install -y h5py"

# run classifiers
su --login ubuntu -c "cd /home/ubuntu/thesis-experiments/workfolder; python eval_LSTM_multitask.py -v -s > eval_LSTM_multitask.log 2>&1"
