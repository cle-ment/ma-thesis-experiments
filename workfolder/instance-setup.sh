#!/bin/bash

# clone repo (replace password)
git clone https://cle-ment:XXXXXXX@github.com/cle-ment/thesis-experiments /home/ubuntu/thesis-experiments

# move data into workfolder
mv /home/ubuntu/2016-07-aws-experiments/workfolder/data /home/ubuntu/thesis-experiments/workfolder/data

# remove old experiments folder
rm -rf /home/ubuntu/2016-07-aws-experiments

# make output dir
mkdir thesis-experiments/output

# copy previous output to continue computation (if available)
scp -r clemens@cwestrup.de:/home/clemens/thesis/output/ /home/ubuntu/thesis-experiments/

# run computation
python /home/ubuntu/thesis-experiments/workfolder/eval_doc2vec.py
