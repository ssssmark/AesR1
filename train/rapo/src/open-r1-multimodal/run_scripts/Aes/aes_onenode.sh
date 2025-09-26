cd src/open-r1-multimodal
unset HCCL_RDMA_TC
unset HCCL_RDMA_SL
export NCCL_CROSS_NIC=1
export NCCL_ALGO=^Ring
export DEBUG_MODE="true"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SHM_DISABLE=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

export NNODES=$ARNOLD_WORKER_NUM
export NODE_RANK=$ARNOLD_ID
export NPROC_PER_NODE=$ARNOLD_WORKER_GPU
export MASTER_PORT=$port
RUN_NAME="qwen2-5-7b_instruct_aescot_1epoch_AesScore_train_15k_RAPO_kl_001"
export LOG_PATH="./log_$RUN_NAME.txt"
export WANDB_PROJECT=AesR1

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/rapo.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir AesR1/models/$RUN_NAME \
    --model_name_or_path models/qwen2-5-7b_instruct_AesCoT_gpt_3k_epoch1 \
    --max_pixels 262144 \
    --question_template scoring \
    --data_file_paths  AesR1/data/aes_train_15k.json \
    --freeze_vision_modules false \
    --use_liger_kernel true \
    --num_generations 4 \
    --max_completion_length 512 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --beta 0.01 \
    --epsilon_high 0.28 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --save_steps 50 \
    --save_only_model true