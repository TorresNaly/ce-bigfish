#!/bin/bash
#SBATCH --account=csu95_alpine1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:30:00
#SBATCH --partition=amilan
#SBATCH --output="logs/%j-report.out"  # Initially save in a default logs directory

# Set the base directory and initialize paths

local_download_directory="/pl/active/onishimura_lab/PROJECTS/naly/bigfish/02_DG_quantification/N2_chs-1_set-3/241114_N2-chs-1/4-cell/input"
input_directory="${local_download_directory}/input"
subdirectories=("$input_directory"/*)
python_script_path="/projects/naly@colostate.edu/bigfish/ce-bigfish/01_Ce-bigIFSH_single-embryo.py"

# Define user-specific and microscope parameters as environment variables
export Cy5="set-3_mRNA"
export mCherry="erm-1_mRNA"
export FITC="noting"
export DAPI="DAPI"
export brightfield="brightfield"
export wavelength_cy5="670"
export wavelength_mCherry="610"
export na="1.42"
export refractive_index_medium="1.515"
export voxel_size="1448,450,450"

# Ensure SLURM_ARRAY_TASK_ID is within bounds of subdirectories array
folder_name="${subdirectories[${SLURM_ARRAY_TASK_ID}]}"

# Check if folder_name is valid
if [[ -d "$folder_name" ]]; then
    # Verify if the folder contains exactly two files: .dv and its reference file
    dv_files=($(find "$folder_name" -maxdepth 1 -type f -name "*.dv" | sort))
    # Check that there are exactly 2 .dv files
    if [[ ${#dv_files[@]} -eq 2 ]] && \
       (echo "${dv_files[@]}" | grep -q "R3D_REF.dv") && \
       (echo "${dv_files[@]}" | grep -q "R3D.dv"); then
        # Create a unique output directory for the current folder
        output_directory="${local_download_directory}/output/$(basename "$folder_name")"
        # Create output folder if it doesn't exist
        if [ ! -d "${output_directory}" ]; then
            mkdir -p "$output_directory"
        fi
        # Set environment variables for the Python script execution
        export FOLDER_NAME="$folder_name"
        export OUTPUT_DIRECTORY="$output_directory"
        
        # Now set the SLURM output file in the script dynamically
        SLURM_OUTPUT="${output_directory}/%j-report.out"  # Dynamic output path for SLURM
        
        # Execute the Python script
        python "$python_script_path" "$folder_name" "$output_directory"
    else
        # Log a message if the folder doesn't contain the required files
        echo "Skipping folder: $folder_name as it does not contain the required files ${dv_files[*]}."
    fi
else
    echo "Invalid or missing directory: $folder_name"
fi

# Wait for all SLURM array tasks to complete before combining CSV files
if [[ ${SLURM_ARRAY_TASK_ID} -eq $((SLURM_ARRAY_TASK_COUNT - 1)) ]]; then
    wait
fi

# Combine all CSVs into a single file
combined_csv="${local_download_directory}/output/combined_quantification.csv"
experiment_csv=()

# Find all individual quantification CSVs in output subdirectories
for csv_file in "${local_download_directory}/output"/*/quantification_*.csv; do
    experiment_csv+=("$csv_file")
done

# Combine CSV files if data is found
if [[ ${#experiment_csv[@]} -gt 0 ]]; then
    # Use pandas in a short Python script to combine CSV files
    python - <<END
import pandas as pd
import glob
# Collect all quantification CSV files
csv_files = glob.glob("${local_download_directory}/output/*/quantification_*.csv")
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
combined_df.to_csv("${combined_csv}", index=False)
print("Combined CSV saved at ${combined_csv}")
END
else
    echo "No quantification data found to combine."
fi

