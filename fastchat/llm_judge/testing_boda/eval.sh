#!/bin/bash

# Load environment variables from parent directory's .env
ENV_FILE="/home/abdelrahman.sadallah/.env"
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

if [ -z "$GPT_KEY_ANWAR" ]; then
    echo "Error: GPT_KEY_ANWAR not set in $ENV_FILE"
    exit 1
fi

export CUDA_LAUNCH_BLOCKING=1
export OPENAI_API_KEY="$GPT_KEY_ANWAR"
export ANTHROPIC_API_KEY="$ANTHROPIC_KEY_BODA"
export TORCH_USE_CUDA_DSA=1

# ================= Configurable parameters =================
LLM_JUDGE_PATH="/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge"

# MODEL_PATH="/mnt/data/users/anwarvic/ONEDRIVE/Projects/jais_plus/checkpoints/20250804_8b_LC_all_exp2p2_6"
# MODEL_ID="jais_8b_LC_all_exp2p2_6"                           # default if not set
MODEL_PATH="ALLaM-AI/ALLaM-7B-Instruct-preview"
MODEL_ID="allam"                # default if not set
BENCH_NAME="vicuna_ar_extended" 
JUDGE_MODEL="claude-sonnet-4-5"               # default if not set
FIRST_N=1
                               # default if not set
# ============================================================

conda activate jais
export CUDA_VISIBLE_DEVICES=0
## important to enable raw tempelate
export VICUNA_ARABIC=1

cd "$LLM_JUDGE_PATH" || { echo "Error: Could not change directory to $LLM_JUDGE_PATH"; exit 1; }
# Step 1: Generate model answers
# echo "Generating model answers..."
# python gen_model_answer.py \
#     --model-path "$MODEL_PATH" \
#     --model-id "$MODEL_ID" \
#     --bench-name "$BENCH_NAME"


MODEL_LIST="allam"
# MODEL_LIST="jais_8b_LC_all_exp2p2_6 allam"                 

MODE="single"    
# MODE="pairwise-all" 



# Step 2: Run evaluation
echo "Running evaluation..."
python "$LLM_JUDGE_PATH/gen_judgment.py" \
    --model-list $MODEL_LIST \
    --mode "$MODE" \
    --bench-name "$BENCH_NAME" \
    --judge-model "$JUDGE_MODEL" \
    --first-n "$FIRST_N"

# # Step 3: Show results
# python "$LLM_JUDGE_PATH/show_result.py" \
#     --model-list $MODEL_LIST \
#     --mode "$MODE" \
#     --bench-name "$BENCH_NAME" 
    
echo "Benchmark evaluation completed successfully."
