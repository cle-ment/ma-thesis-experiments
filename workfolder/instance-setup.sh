#!/bin/bash

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

# copy previous output to continue computation (if available)
su --login ubuntu -c "scp -r clemens@cwestrup.de:/home/clemens/thesis/output/ /home/ubuntu/thesis-experiments/"

# prepare .bash_profile to load enviroment when execiting command as user 'ubuntu'
mv /home/ubuntu/thesis-experiments/workfolder/--.bash_profile /home/ubuntu/.bash_profile

# run computation
su --login ubuntu -c "cd /home/ubuntu/thesis-experiments/workfolder; python eval_doc2vec.py"
