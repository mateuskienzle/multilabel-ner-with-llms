#!/bin/bash
#SBATCH --job-name=qwen3_4b_semclinbr
#SBATCH --partition=tupi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=/home/users/mlkienzle/outputs/%x_%j.out
#SBATCH --error=/home/users/mlkienzle/outputs/%x_%j.err

source ~/envs/tcc_env/bin/activate

RAW=~/results/qwen3_4b_semclinbr_sentence_raw_${SLURM_JOB_ID}

# Zero-shot
OUTPUT_PATH="${RAW}_zero_shot.json"
python3 ~/scripts/run_ner.py \
  --model ~/models/Qwen3-4B \
  --data-dir ~/datasets/SemClinBr-xml-public-v1 \
  --output "$OUTPUT_PATH" \
  --granularity sentence \
 # --max-tokens 16384

# # Few-shot (one-shot)
# OUTPUT_PATH="${RAW}_few_shot.json"
# python3 ~/scripts/run_ner.py \
#   --model ~/models/Qwen3-4B \
#   --data-dir ~/datasets/SemClinBr-xml-public-v1 \
#   --output "$OUTPUT_PATH" \
#   --few-shot \
#   --granularity sentence \
#   # --max-tokens 16384

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --output "${OUTPUT_PATH/raw/metrics}"

python3 ~/scripts/evaluate.py \
  --predictions "$OUTPUT_PATH" \
  --filter parsed_only \
  --output "${OUTPUT_PATH/raw/metrics_parsed_only}"
