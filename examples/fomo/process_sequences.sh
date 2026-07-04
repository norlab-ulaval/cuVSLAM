#!/bin/bash
# process_sequences.sh
# Usage: ./process_sequences.sh <remote_dataset_dir> <local_temp_dir> <output_dir>

REMOTE_DATASET_DIR="/run/user/2002/gvfs/sftp:host=192.168.1.10,user=mabox/FoMo/ijrr"
LOCAL_TEMP_DIR="/tmp/fomo_data"
OUTPUT_DIR="/home/robot/Desktop/output2"

# Array of colors to process. e.g. ("red" "blue" "green" "yellow" "orange" "magenta")
# If empty, no sequences will be processed.
COLORS_TO_PROCESS=("green" "yellow" "magenta")

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

# Iterate over each color sequentially
for current_color in "${COLORS_TO_PROCESS[@]}"; do
    echo "========================================"
    echo "Processing color: $current_color"
    echo "========================================"

    color_output_dir="$OUTPUT_DIR/$current_color"
    mkdir -p "$color_output_dir/processing"
    mkdir -p "$color_output_dir/stats"
    mkdir -p "$color_output_dir/trajectories"

    # Iterate over date directories
    for date_dir in "$REMOTE_DATASET_DIR"/*/; do
        if [ ! -d "$date_dir" ]; then continue; fi
        date_name=$(basename "$date_dir")

        echo "Scanning date folder: $date_name for color: $current_color"

        # Iterate over sequence directories within the date folder
        for seq_dir in "$date_dir"*/; do
            if [ ! -d "$seq_dir" ]; then continue; fi
            seq_name=$(basename "$seq_dir")

            # Filter by current color
            color_prefix="${seq_name%%_*}"
            if [ "$color_prefix" != "$current_color" ]; then
                continue
            fi

            echo "----------------------------------------"
            echo "Processing sequence: $seq_name"

            # 1. Copy locally using rsync
            echo "Copying sequence locally using rsync..."
            local_seq_dir="$LOCAL_TEMP_DIR/$seq_name"
            mkdir -p "$local_seq_dir"
            rsync -a --info=progress2 \
		  --include='zedx_left/***' \
		  --include='zedx_right/***' \
		  --include='calib/***' \
		  --include='vectornav.csv' \
		  --exclude='*' \
		  "$seq_dir" "$local_seq_dir/"

            if [ ! -d "$local_seq_dir" ]; then
                echo "Error: Failed to copy $seq_name locally to $local_seq_dir. Skipping."
                continue
            fi

            proc_dir="$color_output_dir/processing/$seq_name"
            mkdir -p "$proc_dir"

            log_file="$proc_dir/run_slam_${seq_name}.log"
            monitor_json="$color_output_dir/stats/${seq_name}_${seq_name}.json"
            monitor_jpg="$color_output_dir/stats/${seq_name}_${seq_name}.jpg"

            # 2. Start monitor
            echo "Starting system resource monitor..."
            python3 monitor_stats.py --output_json "$monitor_json" --output_jpg "$monitor_jpg" &
            MONITOR_PID=$!

            # 3. Process
            echo "Running SLAM processing (logging to $log_file)..."
            python3 track_fomo.py --slam_sync_mode --sequence_dir "$local_seq_dir" --output_filepath "$proc_dir" --no_vis > "$log_file" 2>&1
            SLAM_EXIT_CODE=$?

            # 4. Stop monitor
            echo "Stopping monitor..."
            kill -SIGTERM $MONITOR_PID
            wait $MONITOR_PID 2>/dev/null

            # 5. Sort trajectory
            traj_src="$proc_dir/trajectory_tum.txt"
            traj_dst="$color_output_dir/trajectories/${seq_name}_${seq_name}.txt"

            if [ -f "$traj_src" ]; then
                mv "$traj_src" "$traj_dst"
                echo "Moved SLAM trajectory to $traj_dst"
            else
                echo "Warning: SLAM trajectory file $traj_src not found!"
            fi

            odom_traj_src="$proc_dir/trajectory_odom_tum.txt"
            odom_traj_dst="$color_output_dir/trajectories/${seq_name}_${seq_name}.txt_bak"

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
            echo "----------------------------------------"
        done
    done
done

echo "Batch processing complete."
