#!/bin/bash
# localize_sequences.sh
# This script localizes specific sequences against a given map.

DEBUG=false
COLOR_FILTER=""

for arg in "$@"; do
    if [ "$arg" == "--debug" ]; then
        DEBUG=true
        echo "=== DEBUG MODE ENABLED - NO COMMANDS WILL BE EXECUTED ==="
    elif [ "$arg" == "orange" ] || [ "$arg" == "yellow" ]; then
        COLOR_FILTER="$arg"
    fi
done

REMOTE_DATASET_DIR="$HOME/fomo/ijrr"
LOCAL_TEMP_DIR="$HOME/Desktop/fomo_data_loc"
OUTPUT_DIR="$HOME/Desktop/output_localization"

if [ ! -d "$REMOTE_DATASET_DIR" ]; then
    echo "Error: Remote dataset dir not found at $REMOTE_DATASET_DIR"
    exit 1
fi

# Array of target sequences in format "date_folder:map_sequence:eval_sequence"
SEQUENCES=(
    "2025-01-29:orange_2025-01-30-09-07:orange_2025-01-30-09-07"
    "2025-01-29:yellow_2025-01-29-16-08:yellow_2025-01-29-16-08"
    "2025-06-26:orange_2025-06-26-10-48:orange_2025-06-26-10-48"
    "2025-06-26:yellow_2025-06-26-14-42:yellow_2025-06-26-14-42"
    "2025-10-14:orange_2025-10-14-12-46:orange_2025-10-14-12-46"
    "2025-10-14:yellow_2025-10-14-14-48:yellow_2025-10-14-14-48"
)

if [ "$DEBUG" = true ]; then
    echo "mkdir -p $LOCAL_TEMP_DIR"
    echo "mkdir -p $OUTPUT_DIR"
else
    mkdir -p "$LOCAL_TEMP_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

for entry in "${SEQUENCES[@]}"; do
    IFS=':' read -r date_folder map_sequence eval_sequence <<< "$entry"
    
    color="${map_sequence%%_*}"
    
    # Filter by color if requested
    if [ -n "$COLOR_FILTER" ] && [ "$color" != "$COLOR_FILTER" ]; then
        continue
    fi

    echo "================================================="
    echo "Processing date: $date_folder"
    echo "Evaluating sequence: $eval_sequence"
    echo "Using map from: $map_sequence"
    echo "================================================="

    seq_dir="$REMOTE_DATASET_DIR/$date_folder/$eval_sequence"
    if [ ! -d "$seq_dir" ]; then
        echo "Error: Sequence directory $seq_dir not found. Skipping."
        continue
    fi

    # 1. Copy sequence locally using rsync
    echo "Copying sequence locally using rsync..."
    local_seq_dir="$LOCAL_TEMP_DIR/$eval_sequence"
    
    if [ "$DEBUG" = true ]; then
        echo "mkdir -p $local_seq_dir"
        echo "rsync -a --info=progress2 --include='zedx_left/***' --include='zedx_right/***' --include='calib/***' --include='vectornav.csv' --exclude='*' $seq_dir/ $local_seq_dir/"
    else
        mkdir -p "$local_seq_dir"
        rsync -a --info=progress2 \
              --include='zedx_left/***' \
              --include='zedx_right/***' \
              --include='calib/***' \
              --include='vectornav.csv' \
              --exclude='*' \
              "$seq_dir/" "$local_seq_dir/"
              
        if [ ! -d "$local_seq_dir" ]; then
            echo "Error: Failed to copy $eval_sequence locally. Skipping."
            continue
        fi
    fi

    # 2. Setup output directory and copy map
    proc_dir="$OUTPUT_DIR/$eval_sequence"
    
    MAPS_DIR="$HOME/fomo/evaluation/$color/pycuvslam_no_imu.offline/processing"
    
    map_src="$MAPS_DIR/$map_sequence/map.mdb"
    map_dst="$proc_dir/map/map.mdb"

    echo "Copying map from $map_src to $map_dst..."
    if [ ! -f "$map_src" ]; then
        echo "Error: Map file not found at $map_src. Skipping."
        continue
    fi
    
    if [ "$DEBUG" = true ]; then
        echo "mkdir -p $proc_dir/map"
        echo "cp $map_src $map_dst"
    else
        mkdir -p "$proc_dir/map"
        cp "$map_src" "$map_dst"
    fi

    log_file="$proc_dir/run_loc_${eval_sequence}.log"

    # 3. Process with localization
    echo "Running SLAM localization (logging to $log_file)..."
    if [ "$DEBUG" = true ]; then
        echo "uv run python track_fomo_slam.py --sequence_dir $local_seq_dir --output_filepath $proc_dir --localize --no_vis > $log_file 2>&1"
        SLAM_EXIT_CODE=0
    else
        uv run python track_fomo_slam.py --sequence_dir "$local_seq_dir" --output_filepath "$proc_dir" --localize --no_vis > "$log_file" 2>&1
        SLAM_EXIT_CODE=$?
    fi

    # Plot trajectories
    TRAJ_DIR="$HOME/fomo/evaluation/$color/pycuvslam_no_imu.offline/trajectories"
    slam_traj="$TRAJ_DIR/${map_sequence}_${map_sequence}.txt"
    echo "Plotting trajectories..."
    if [ "$DEBUG" = true ]; then
        echo "uv run python plot_trajectories.py --sequence_dir $proc_dir --slam_traj $slam_traj"
    else
        uv run python plot_trajectories.py --sequence_dir "$proc_dir" --slam_traj "$slam_traj"
    fi

    # 4. Verify and cleanup
    if [ $SLAM_EXIT_CODE -eq 0 ]; then
        echo "Localization successful. Cleaning up local copy of sequence..."
        if [ "$DEBUG" = true ]; then
            echo "rm -rf $local_seq_dir"
        else
            rm -rf "$local_seq_dir"
        fi
    else
        echo "Error: Localization failed with exit code $SLAM_EXIT_CODE. Keeping local copy at $local_seq_dir for debugging."
    fi

    echo "Finished evaluating sequence: $eval_sequence"
    echo "-------------------------------------------------"
done

echo "Batch localization processing complete."
