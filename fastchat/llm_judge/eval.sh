#!/bin/bash

# Load environment variables from parent directory's .env
# ENV_FILE="../.env"
# if [ -f "$ENV_FILE" ]; then
#     export $(grep -v '^#' "$ENV_FILE" | xargs)
# fi

# if [ -z "$GPT_KEY_ANWAR" ]; then
#     echo "Error: GPT_KEY_ANWAR not set in $ENV_FILE"
#     exit 1
# fi

# ================= Configurable parameters =================
LLM_JUDGE_PATH="/home/abdelrahman.sadallah/mbzuai/FastChat/fastchat/llm_judge"

MODEL_PATH="/mnt/data/users/anwarvic/ONEDRIVE/Projects/jais_plus/checkpoints/20250804_8b_LC_all_exp2p2_6"    # default if not set
MODEL_ID="jais_8b_LC_all_exp2p2_6"                            # default if not set
BENCH_NAME="vicuna_bench"                # default if not set

                               # default if not set
# ============================================================

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
conda activate jais
export CUDA_VISIBLE_DEVICES=1

cd "$LLM_JUDGE_PATH" || { echo "Error: Could not change directory to $LLM_JUDGE_PATH"; exit 1; }
# Step 1: Generate model answers
echo "Generating model answers..."
python gen_model_answer.py \
    --model-path "$MODEL_PATH" \
    --model-id "$MODEL_ID" \
    --bench-name "$BENCH_NAME"


MODEL_LIST="${MODEL_LIST:-FANAR ALLaM}"                 # default if not set
MODE="${MODE:-pairwise-all}"                             # default if not set
FIRST_N="${FIRST_N:-5}"   
    # # Step 2: Run evaluation
    # echo "Running evaluation..."
    # python "$LLM_JUDGE_PATH/gen_judgment.py" \
    #     --model-list $MODEL_LIST \
    #     --mode "$MODE" \
    #     --bench-name "$BENCH_NAME" \
    #     --first-n "$FIRST_N"

    # # python show_result.py --model-list FANAR ALLaM --bench-name vicuna_bench --mode pairwise-all
    # echo "Benchmark evaluation completed successfully."
