# !/bin/bash

echo -n "Enter participant name: "
read participant_name
trials_folder="$(date +"%d-%b-%y")_$participant_name"

dest="$(pwd)/analysis/StimuliVerificationTrials/$trials_folder"

mkdir -p "$dest"

mv "$(pwd)/$participant_name"*.log "$dest"
mv "$(pwd)/$participant_name"*.gdf "$dest"