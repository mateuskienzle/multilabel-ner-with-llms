#!/bin/bash
#SBATCH --job-name=gemma_harem
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/mlkienzle/outputs/%x_%j.out
#SBATCH --error=/home/users/mlkienzle/outputs/%x_%j.err

source ~/envs/tcc_env/bin/activate

RAW=~/results/gemma_harem_sentence_raw_${SLURM_JOB_ID}

# Zero-shot
OUTPUT_PATH="${RAW}_zero_shot.json"
python3 ~/scripts/run_ner.py \
  --model ~/models/gemma-3-4b-it \
  --data-dir ~/datasets/Harem \
  --output "$OUTPUT_PATH" \
  --dataset harem \
  --max-tokens 4096 \
  --max-model-len 8192

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --output "${OUTPUT_PATH/raw/metrics}"

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --filter parsed_only \
  --output "${OUTPUT_PATH/raw/metrics_parsed_only}"

# Few-shot (one-shot)
OUTPUT_PATH="${RAW}_few_shot.json"
python3 ~/scripts/run_ner.py \
  --model ~/models/gemma-3-4b-it \
  --data-dir ~/datasets/Harem \
  --output "$OUTPUT_PATH" \
  --dataset harem \
  --few-shot \
  --max-tokens 4096 \
  --max-model-len 8192

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --output "${OUTPUT_PATH/raw/metrics}"

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --filter parsed_only \
  --output "${OUTPUT_PATH/raw/metrics_parsed_only}"
