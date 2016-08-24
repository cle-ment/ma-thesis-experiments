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

# generate feature spaces (will skip if they already exist)
su --login ubuntu -c "cd /home/ubuntu/thesis-experiments/workfolder; python generate_best_feature_spaces.py"

# run classifiers
su --login ubuntu -c "cd /home/ubuntu/thesis-experiments/workfolder; THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python eval_classifiers_nns.py > eval_classifiers_nns.log 2>&1"
