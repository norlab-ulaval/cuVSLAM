#!/bin/bash
# process_sequences.sh
# Usage: ./process_sequences.sh <remote_dataset_dir> <local_temp_dir> <output_dir>

REMOTE_DATASET_DIR=$1
LOCAL_TEMP_DIR=$2
OUTPUT_DIR=$3

# Array of colors to process. e.g. ("red" "blue" "green" "yellow" "orange" "magenta")
# If empty, no sequences will be processed.
COLORS_TO_PROCESS=("red" "blue" "green" "yellow" "orange" "magenta")

if [ -z "$REMOTE_DATASET_DIR" ] || [ -z "$LOCAL_TEMP_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: ./process_sequences.sh <remote_dataset_dir> <local_temp_dir> <output_dir>"
    exit 1
fi

# Activate the virtual environment
source /home/robot/Desktop/cuVSLAM/.venv/bin/activate

if [ ${#COLORS_TO_PROCESS[@]} -eq 0 ]; then
    echo "Warning: COLORS_TO_PROCESS is empty. No sequences will be processed."
    exit 0
fi

mkdir -p "$OUTPUT_DIR/processing"
mkdir -p "$OUTPUT_DIR/stats"
mkdir -p "$OUTPUT_DIR/trajectories"

# Iterate over date directories
for date_dir in "$REMOTE_DATASET_DIR"/*/; do
    if [ ! -d "$date_dir" ]; then continue; fi
    date_name=$(basename "$date_dir")
    
    echo "Scanning date folder: $date_name"
    
    # Iterate over sequence directories within the date folder
    for seq_dir in "$date_dir"*/; do
        if [ ! -d "$seq_dir" ]; then continue; fi
        seq_name=$(basename "$seq_dir")
        
        # Filter by color
        color_prefix="${seq_name%%_*}"
        match=0
        for c in "${COLORS_TO_PROCESS[@]}"; do
            if [ "$c" = "$color_prefix" ]; then
                match=1
                break
            fi
        done
        if [ $match -eq 0 ]; then
            continue
        fi
        
        echo "========================================"
        echo "Processing sequence: $seq_name"
        
        # 1. Copy locally using rsync
        echo "Copying sequence locally using rsync..."
        local_seq_dir="$LOCAL_TEMP_DIR/$seq_name"
        mkdir -p "$local_seq_dir"
        rsync -a --info=progress2 "$seq_dir" "$local_seq_dir/"
        
        if [ ! -d "$local_seq_dir" ]; then
            echo "Error: Failed to copy $seq_name locally to $local_seq_dir. Skipping."
            continue
        fi
        
        proc_dir="$OUTPUT_DIR/processing/$seq_name"
        mkdir -p "$proc_dir"
        
        log_file="$proc_dir/run_slam_${seq_name}.log"
        monitor_json="$OUTPUT_DIR/stats/${seq_name}_${seq_name}.json"
        monitor_jpg="$OUTPUT_DIR/stats/${seq_name}_${seq_name}.jpg"
        
        # 2. Start monitor
        echo "Starting system resource monitor..."
        python3 monitor_stats.py --output_json "$monitor_json" --output_jpg "$monitor_jpg" &
        MONITOR_PID=$!
        
        # 3. Process
        echo "Running SLAM processing (logging to $log_file)..."
        python3 track_fomo_slam.py --slam_sync_mode --sequence_dir "$local_seq_dir" --output_filepath "$proc_dir" --no_vis > "$log_file" 2>&1
        SLAM_EXIT_CODE=$?
        
        # 4. Stop monitor
        echo "Stopping monitor..."
        kill -SIGTERM $MONITOR_PID
        wait $MONITOR_PID 2>/dev/null
        
        # 5. Sort trajectory
        traj_src="$proc_dir/trajectory_tum.txt"
        traj_dst="$OUTPUT_DIR/trajectories/${seq_name}_${seq_name}.txt"
        
        if [ -f "$traj_src" ]; then
            mv "$traj_src" "$traj_dst"
            echo "Moved SLAM trajectory to $traj_dst"
        else
            echo "Warning: SLAM trajectory file $traj_src not found!"
        fi

        odom_traj_src="$proc_dir/trajectory_odom_tum.txt"
        odom_traj_dst="$OUTPUT_DIR/trajectories/${seq_name}_${seq_name}.txt_bak"
        
        if [ -f "$odom_traj_src" ]; then
            mv "$odom_traj_src" "$odom_traj_dst"
            echo "Moved Odom trajectory to $odom_traj_dst"
        else
            echo "Warning: Odom trajectory file $odom_traj_src not found!"
        fi
        
        # 6. Verify and cleanup
        if [ $SLAM_EXIT_CODE -eq 0 ]; then
            echo "Processing successful. Cleaning up local copy..."
            rm -rf "$local_seq_dir"
        else
            echo "Error: Processing failed with exit code $SLAM_EXIT_CODE. Keeping local copy at $local_seq_dir for debugging."
        fi
        
        echo "Finished sequence: $seq_name"
        echo "========================================"
    done
done

echo "Batch processing complete."
