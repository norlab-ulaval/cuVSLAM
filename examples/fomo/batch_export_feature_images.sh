#!/bin/bash

# Default values
DATASET_BASE_PATH=${1:-"/path/to/dataset"}
OUTPUT_BASE_PATH=${2:-"./feature_outputs"}
SCRIPT_PATH="examples/fomo/export_feature_images.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH not found. Please run this script from the root of cuVSLAM."
    exit 1
fi

echo "Using Dataset Base Path: $DATASET_BASE_PATH"
echo "Using Output Base Path: $OUTPUT_BASE_PATH"

mkdir -p "$OUTPUT_BASE_PATH"

# Mapping of Recording Name -> Deployment Date Directory
declare -A RECORDINGS=(
    ["orange_2025-01-10-09-45"]="2025-01-10"
    ["orange_2025-01-30-09-07"]="2025-01-29"
    ["orange_2025-03-14-09-07"]="2025-03-10"
    ["yellow_2024-11-28-09-44"]="2024-11-28"
    ["yellow_2025-01-29-16-08"]="2025-01-29"
    ["yellow_2025-03-10-17-22"]="2025-03-10"
    ["orange_2025-06-26-10-48"]="2025-06-26"
    ["orange_2025-08-20-13-17"]="2025-08-20"
    ["orange_2025-09-24-11-51"]="2025-09-24"
    ["yellow_2025-06-26-14-42"]="2025-06-26"
    ["yellow_2025-08-20-10-54"]="2025-08-20"
    ["yellow_2025-09-24-11-12"]="2025-09-24"
)

# Order to process them in (to maintain consistent execution order)
RECORDING_ORDER=(
    "orange_2025-01-10-09-45"
    "orange_2025-01-30-09-07"
    "orange_2025-03-14-09-07"
    "yellow_2024-11-28-09-44"
    "yellow_2025-01-29-16-08"
    "yellow_2025-03-10-17-22"
    "orange_2025-06-26-10-48"
    "orange_2025-08-20-13-17"
    "orange_2025-09-24-11-51"
    "yellow_2025-06-26-14-42"
    "yellow_2025-08-20-10-54"
    "yellow_2025-09-24-11-12"
)

for recording in "${RECORDING_ORDER[@]}"; do
    deployment_date="${RECORDINGS[$recording]}"
    search_dir="$DATASET_BASE_PATH/$deployment_date"
    
    echo "--------------------------------------------------------"
    echo "Processing recording: $recording"
    
    if [ ! -d "$search_dir" ]; then
        echo "Warning: Deployment directory $search_dir not found. Skipping."
        continue
    fi

    # Find the full trajectory directory containing the recording name
    # Using find to get the first matching directory
    sequence_dir=$(find "$search_dir" -maxdepth 1 -type d -name "*$recording*" | head -n 1)

    if [ -z "$sequence_dir" ] || [ ! -d "$sequence_dir" ]; then
        echo "Warning: Trajectory directory for $recording not found in $search_dir. Skipping."
        continue
    fi
    
    echo "Found sequence directory: $sequence_dir"
    
    # Create specific output folder for this trajectory
    output_filepath="$OUTPUT_BASE_PATH/$recording"
    mkdir -p "$output_filepath"
    
    echo "Output will be saved to: $output_filepath"
    
    # Run the feature export script
    # --no_vis is passed so it doesn't try to open rerun instances sequentially for each dataset
    cmd="python \"$SCRIPT_PATH\" --sequence_dir \"$sequence_dir\" --output_filepath \"$output_filepath\" --no_vis"
    echo "Running: $cmd"
    
    # Execute the command
    eval "$cmd"
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed $recording"
    else
        echo "Error: Processing failed for $recording"
    fi
done

echo "--------------------------------------------------------"
echo "All done!"
