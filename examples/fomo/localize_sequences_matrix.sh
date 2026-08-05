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
OUTPUT_DIR="$HOME/Desktop/output_localization_matrix_fixed"

if [ ! -d "$REMOTE_DATASET_DIR" ]; then
    echo "Error: Remote dataset dir not found at $REMOTE_DATASET_DIR"
    exit 1
fi

# Array of target sequences in format "date_folder:map_sequence:eval_sequence"
SEQUENCES=(
    # Orange map 1 (Jan)
    "2024-11-21:orange_2025-01-30-09-07:orange_2024-11-21-11-10"
    "2025-03-10:orange_2025-01-30-09-07:orange_2025-03-14-09-07"
    "2025-08-20:orange_2025-01-30-09-07:orange_2025-08-20-13-17"
    # Orange map 2 (Jun)
    "2024-11-21:orange_2025-06-26-10-48:orange_2024-11-21-11-10"
    "2025-03-10:orange_2025-06-26-10-48:orange_2025-03-14-09-07"
    "2025-08-20:orange_2025-06-26-10-48:orange_2025-08-20-13-17"
    # Orange map 3 (Oct)
    "2024-11-21:orange_2025-10-14-12-46:orange_2024-11-21-11-10"
    "2025-03-10:orange_2025-10-14-12-46:orange_2025-03-14-09-07"
    "2025-08-20:orange_2025-10-14-12-46:orange_2025-08-20-13-17"

    # Yellow map 1 (Jan)
    "2024-11-21:yellow_2025-01-29-16-08:yellow_2024-11-21-14-26"
    "2025-03-10:yellow_2025-01-29-16-08:yellow_2025-03-10-17-22"
    "2025-08-20:yellow_2025-01-29-16-08:yellow_2025-08-20-10-54"
    # Yellow map 2 (Jun)
    "2024-11-21:yellow_2025-06-26-14-42:yellow_2024-11-21-14-26"
    "2025-03-10:yellow_2025-06-26-14-42:yellow_2025-03-10-17-22"
    "2025-08-20:yellow_2025-06-26-14-42:yellow_2025-08-20-10-54"
    # Yellow map 3 (Oct)
    "2024-11-21:yellow_2025-10-14-14-48:yellow_2024-11-21-14-26"
    "2025-03-10:yellow_2025-10-14-14-48:yellow_2025-03-10-17-22"
    "2025-08-20:yellow_2025-10-14-14-48:yellow_2025-08-20-10-54"
)

if [ "$DEBUG" = true ]; then
    echo "mkdir -p $LOCAL_TEMP_DIR"
    echo "mkdir -p $OUTPUT_DIR"
else
    mkdir -p "$LOCAL_TEMP_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# ---------------------------------------------------------------------------
# Build a grouped structure: for each unique (date_folder, eval_sequence) pair,
# collect all map_sequences that apply to it (respecting the color filter).
# We use associative arrays keyed by eval_sequence, preserving insertion order
# via a separate GROUP_ORDER array.
# ---------------------------------------------------------------------------
declare -A GROUP_DATE      # GROUP_DATE[eval_seq]  = date_folder
declare -A GROUP_MAPS      # GROUP_MAPS[eval_seq]  = "map1 map2 map3 ..." (space-separated)
declare -a GROUP_ORDER     # Insertion-order list of eval_seq keys

for entry in "${SEQUENCES[@]}"; do
    IFS=':' read -r date_folder map_sequence eval_sequence <<< "$entry"

    color="${map_sequence%%_*}"

    # Apply color filter
    if [ -n "$COLOR_FILTER" ] && [ "$color" != "$COLOR_FILTER" ]; then
        continue
    fi

    if [ -z "${GROUP_DATE[$eval_sequence]+_}" ]; then
        GROUP_DATE[$eval_sequence]="$date_folder"
        GROUP_MAPS[$eval_sequence]="$map_sequence"
        GROUP_ORDER+=("$eval_sequence")
    else
        GROUP_MAPS[$eval_sequence]="${GROUP_MAPS[$eval_sequence]} $map_sequence"
    fi
done

# ---------------------------------------------------------------------------
# Process each unique eval_sequence once: copy once, run all maps, then clean up.
# ---------------------------------------------------------------------------
for eval_sequence in "${GROUP_ORDER[@]}"; do
    date_folder="${GROUP_DATE[$eval_sequence]}"
    maps_str="${GROUP_MAPS[$eval_sequence]}"
    read -ra map_list <<< "$maps_str"

    color="${eval_sequence%%_*}"

    echo "================================================="
    echo "Processing eval sequence : $eval_sequence"
    echo "Date folder              : $date_folder"
    echo "Maps to evaluate against : ${map_list[*]}"
    echo "================================================="

    seq_dir="$REMOTE_DATASET_DIR/$date_folder/$eval_sequence"
    if [ ! -d "$seq_dir" ]; then
        echo "Error: Sequence directory $seq_dir not found. Skipping."
        continue
    fi

    # ------------------------------------------------------------------
    # 1. Copy sequence locally (once for all maps)
    # ------------------------------------------------------------------
    local_seq_dir="$LOCAL_TEMP_DIR/$eval_sequence"
    COPIED_DATA=false

    if [ -f "$local_seq_dir/.copy_done" ]; then
        echo "Sequence data already exists locally at $local_seq_dir. Skipping copy."
    else
        echo "Copying sequence locally using rsync..."
        COPIED_DATA=true
        if [ "$DEBUG" = true ]; then
            echo "rm -rf $local_seq_dir"
            echo "mkdir -p $local_seq_dir"
            echo "rsync -a --info=progress2 --include='zedx_left/***' --include='zedx_right/***' --include='calib/***' --include='vectornav.csv' --exclude='*' $seq_dir/ $local_seq_dir/"
            echo "find $local_seq_dir -type f -name \".*\" -delete 2>/dev/null || true"
            echo "left_count=\$(find \"$local_seq_dir/zedx_left\" -maxdepth 1 -type f -not -name \".*\" 2>/dev/null | wc -l)"
            echo "right_count=\$(find \"$local_seq_dir/zedx_right\" -maxdepth 1 -type f -not -name \".*\" 2>/dev/null | wc -l)"
            echo "if [ \"\$left_count\" -ne \"\$right_count\" ] || [ \"\$left_count\" -eq 0 ]; then echo 'File count mismatch or zero files. Skipping all maps for this sequence.'; continue; fi"
            echo "touch $local_seq_dir/.copy_done"
        else
            rm -rf "$local_seq_dir"
            mkdir -p "$local_seq_dir"
            rsync -a --info=progress2 \
                  --include='zedx_left/***' \
                  --include='zedx_right/***' \
                  --include='calib/***' \
                  --include='vectornav.csv' \
                  --exclude='*' \
                  "$seq_dir/" "$local_seq_dir/"

            if [ $? -ne 0 ]; then
                echo "Error: Failed to copy $eval_sequence locally. Skipping all maps for this sequence."
                continue
            fi

            # Remove any hidden temp files (starting with .) that might break image loading
            find "$local_seq_dir" -type f -name ".*" -delete 2>/dev/null || true

            # Verify file counts
            left_count=$(find "$local_seq_dir/zedx_left" -maxdepth 1 -type f -not -name ".*" 2>/dev/null | wc -l)
            right_count=$(find "$local_seq_dir/zedx_right" -maxdepth 1 -type f -not -name ".*" 2>/dev/null | wc -l)

            if [ "$left_count" -ne "$right_count" ] || [ "$left_count" -eq 0 ]; then
                echo "Error: File count mismatch (left: $left_count, right: $right_count) or zero files. Skipping all maps for this sequence."
                continue
            fi

            touch "$local_seq_dir/.copy_done"
        fi
    fi

    # ------------------------------------------------------------------
    # 2. Run localization against every map for this eval sequence
    # ------------------------------------------------------------------
    ANY_FAILURE=false

    for map_sequence in "${map_list[@]}"; do
        MAPS_DIR="$HOME/fomo/evaluation/$color/pycuvslam_no_imu.offline/processing"
        TRAJECTORIES_DIR="$HOME/fomo/evaluation/$color/pycuvslam_no_imu.offline/trajectories"

        proc_dir="$OUTPUT_DIR/${eval_sequence}_map_${map_sequence}"
        map_src="$MAPS_DIR/$map_sequence/map.mdb"
        map_dst="$proc_dir/map/data.mdb"
        traj_src="$TRAJECTORIES_DIR/${map_sequence}_${map_sequence}.txt"
        traj_dst="$proc_dir/trajectory_tum.txt"
        log_file="$proc_dir/run_loc_${eval_sequence}.log"

        echo "-------------------------------------------------"
        echo "Map: $map_sequence"

        if [ ! -f "$map_src" ]; then
            echo "Error: Map file not found at $map_src. Skipping this map."
            ANY_FAILURE=true
            continue
        fi
        if [ ! -f "$traj_src" ]; then
            echo "Error: Trajectory file not found at $traj_src. Skipping this map."
            ANY_FAILURE=true
            continue
        fi

        echo "Copying map and trajectory to $proc_dir..."
        if [ "$DEBUG" = true ]; then
            echo "mkdir -p $proc_dir/map"
            echo "cp $map_src $map_dst"
            echo "cp $traj_src $traj_dst"
        else
            mkdir -p "$proc_dir/map"
            cp "$map_src" "$map_dst"
            cp "$traj_src" "$traj_dst"
        fi

        echo "Running SLAM localization (logging to $log_file)..."
        if [ "$DEBUG" = true ]; then
            echo "uv run python track_fomo_slam.py --sequence_dir $local_seq_dir --output_filepath $proc_dir --localize --no_vis"
            SLAM_EXIT_CODE=0
        else
            uv run python track_fomo_slam.py \
                --sequence_dir "$local_seq_dir" \
                --output_filepath "$proc_dir" \
                --localize --no_vis
            SLAM_EXIT_CODE=$?
        fi

        # Plot trajectories
        slam_traj="$TRAJECTORIES_DIR/${map_sequence}_${map_sequence}.txt"
        echo "Plotting trajectories..."
        if [ "$DEBUG" = true ]; then
            echo "uv run python plot_trajectories.py --sequence_dir $proc_dir --slam_traj $slam_traj"
        else
            uv run python plot_trajectories.py \
                --sequence_dir "$proc_dir" \
                --slam_traj "$slam_traj"
        fi

        if [ $SLAM_EXIT_CODE -ne 0 ]; then
            echo "Error: Localization failed (exit code $SLAM_EXIT_CODE) for map $map_sequence."
            ANY_FAILURE=true
        else
            echo "Localization successful for map: $map_sequence"
        fi
    done

    # ------------------------------------------------------------------
    # 3. Cleanup: remove local copy only if it was freshly copied and
    #    all map runs succeeded.
    # ------------------------------------------------------------------
    
    echo "Finished all maps for eval sequence: $eval_sequence"
    echo "================================================="
done

echo "Batch localization processing complete."
