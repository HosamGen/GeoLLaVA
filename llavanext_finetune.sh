#! /bin/bash

#SBATCH --job-name=eval6ep # Job name
#SBATCH --output=/home/hosam.elgendy/output_.%A.txt # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=40G # Total RAM to be used
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH --gres=gpu:1 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for your job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs

# Load your local CUDA environment
export CUDA_HOME=/home/hosam.elgendy/cuda  # Specify your local CUDA path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate your Conda environment
source /home/hosam.elgendy/miniconda3/bin/activate test  # Adjust this to your Conda environment path and name

cd GeoLLaVA/

# Run the first script with flags
# python /home/hosam.elgendy/GeoLLaVA/llavanext_finetune.py --channel --model_type sample --batch_size 2 --prune_amount 0.80 --epochs 6

# Run the second script with flag
python /home/hosam.elgendy/GeoLLaVA/llavanext_eval.py --model_path "/home/hosam.elgendy/GeoLLaVA/outputs/prune_chnl_20.0_LLaVA-NeXT-Video-7B-hf_sample_QLORA_4bit_r64_alpha128_6epochs"
