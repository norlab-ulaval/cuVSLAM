#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mbo/pycuVSLAM/.venv/lib/python3.10/site-packages/nvidia/cublas/lib/
source .venv/bin/activate

input_path_host=/home/mbo/bigfoot-FoMo/ijrr

process_trajectory() {
    local trajectory=$1
    local output_path_host="/home/mbo/output/pycuVSLAM-offline/pycuVSLAM-${trajectory}"
    for date_dir in "${input_path_host}"/*/; do
        date=$(basename "${date_dir}")
        if [[ ${date} == "2024-11-21" || ${date} == "2024-11-28" || ${date} == "2025-01-29" || ${date} == "2025-03-10" || ${date} == "2025-06-26" ]]; then
            continue
        fi
        for dataset_dir in "${date_dir}${trajectory}_"*/; do
            [ -d "${dataset_dir}" ] || continue
            dataset=$(basename "${dataset_dir}")
            echo $dataset
            mkdir -p "${output_path_host}/${date}/${dataset}"
            echo "Processing dataset ${date}/${dataset}"

            log_file="${output_path_host}/${date}/${dataset}/output.log"
            echo "START_TIME: $(date +%s)" > "$log_file"

            python3 -u examples/fomo/track_custom.py \
                --data "${dataset_dir}"\
                --map-out "${output_path_host}/${date}/${dataset}/map_db.cuvslam" \
                --traj-out "${output_path_host}/${date}/${dataset}/${dataset}_${dataset}.txt" \
                2>&1 | tee -a "$log_file"

            echo "END_TIME: $(date +%s)" >> "$log_file"
        done
    done
}

# process_trajectory "red"
# process_trajectory "blue"
# process_trajectory "orange"
# process_trajectory "green"
# process_trajectory "magenta"
process_trajectory "yellow"
