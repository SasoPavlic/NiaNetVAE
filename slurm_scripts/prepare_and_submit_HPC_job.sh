#!/bin/bash

# Delete file nianetvae.sh if it exists
rm -f nianetvae.sh

# wget file from URL into the current folder
wget https://github.com/SasoPavlic/NiaNetVAE/raw/main/nianetvae.sh

# Change permissions to 777
chmod 777 nianetvae.sh

# Delete folders logs, data, configs if they exist
rm -rf logs data configs

# Create folders with permissions 777
mkdir -m 777 logs data configs

# Check if data.zip exists, otherwise download it from URL
if [ ! -f data.zip ]; then
    wget https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip -O data.zip
fi

# Unzip the content of data.zip to the folder data
unzip data.zip -d data

# wget file from URL into config folder
wget https://github.com/SasoPavlic/NiaNetVAE/raw/main/configs/main_config.yaml -P configs

# Check if parameter is passed to the script
if [ -z "$1" ]; then
    echo "No argument supplied"
    echo "Usage: ./start.sh <number_of_runs>"
    echo "Example: ./start.sh 10"
    exit 1
else
    # Run the job N times based on the parameter passed into this script
    for i in $(seq 1 $1); do
        # Run nianetvae.sh
        sbatch nianetvae.sh
    done
fi
