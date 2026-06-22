#!/bin/bash
#SBATCH --job-name=tucano_harem
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/mlkienzle/outputs/%x_%j.out
#SBATCH --error=/home/users/mlkienzle/outputs/%x_%j.err

source ~/envs/tcc_env/bin/activate

RAW=~/results/tucano_harem_sentence_raw_${SLURM_JOB_ID}

# Zero-shot only — few-shot is not feasible for Tucano2-3.7B due to its 4096-token context limit
OUTPUT_PATH="${RAW}_zero_shot.json"
python3 ~/scripts/run_ner.py \
  --model ~/models/Tucano2-qwen-3.7B-Instruct \
  --data-dir ~/datasets/Harem \
  --output "$OUTPUT_PATH" \
  --dataset harem \
  --max-tokens 4096

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --output "${OUTPUT_PATH/raw/metrics}"

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --filter parsed_only \
  --output "${OUTPUT_PATH/raw/metrics_parsed_only}"
