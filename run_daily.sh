#!/bin/bash

# Exit if any command fails
set -e

# Activate conda environment
echo "Activating conda environment 'daily_sca'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate daily_sca

# Path to your Python script and config
SCRIPT_PATH="./main_new.py"
CONFIG_PATH="./config.json"

for year in 2223 1314 1415 1516 1617 1718 1819 1920 2021 2122 ; do
  new_val="hy${year}"
  for catch in Area06; do
    jq --arg v "$new_val" --arg c "$catch" '
      .hy_xxxx = $v
      | .catchment = $c
      | .HR_directories.Sentinel2  |= sub("Area[0-9]+"; $c)
      | .HR_directories.Landsat8  |= sub("Area[0-9]+"; $c)
      | .HR_directories.Landsat9  |= sub("Area[0-9]+"; $c)
      | .LR_directory              |= sub("Area[0-9]+"; $c)
      | .outfld                    |= sub("Area[0-9]+"; $c)
      | .DEM_path                  |= sub("Area[0-9]+"; $c)
    ' "$CONFIG_PATH" > tmp.json && mv tmp.json "$CONFIG_PATH"

    echo "Updated config.json with hy_xxxx=$new_val and catchment=$catch"
    echo "Running the daily SCA script..."
    python "$SCRIPT_PATH" "$CONFIG_PATH"
    
  done
done


echo "Done DK."

