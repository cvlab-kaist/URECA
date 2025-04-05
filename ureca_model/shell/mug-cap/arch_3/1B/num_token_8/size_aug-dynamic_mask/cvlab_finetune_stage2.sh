wandb login 576d24b59b99565351471bc203b22bbe7f214e3d

set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=36300
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

ARCH_NAME="ARCH_3"
GROUP_NAME="${ARCH_NAME}/1B-num_mask8-size_aug-dynamic_mask"
GROUP_NAME="${GROUP_NAME}/stage_2"
OUTPUT_DIR="/scratch/slurm-user24-kaist/sangbeom/media/training/a100/${GROUP_NAME}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
# --------------------------------------------------------- #
# Stage 2 trains only llm lora layer + freeze mlp

CUDA_VISIBLE_DEVICES='2,3' /scratch/slurm-user24-kaist/miniconda3/envs/mug-cap/bin/torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/mug_finetune_with_test.py \
  --model_name_or_path "/scratch/slurm-user24-kaist/sangbeom/media/training/a100/ARCH_3/1B-num_mask8-size_aug-dynamic_mask/stage_1/checkpoint-8635" \
  --conv_style "mug-cap" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/scratch/slurm-user24-kaist/sangbeom/iccv/MUG-CAP/internvl_chat/shell/data/mug-cap_a100.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --freeze_mask_mlp True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --eval_accumulation_steps 1 \
  --evaluation_strategy "steps" \
  --eval_steps 2150 \
  --save_strategy "steps" \
  --save_steps 2150 \
  --save_total_limit 10 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "wandb" \
  --group_name "${GROUP_NAME}" \
  --num_mask_token 8 \
  --testset_size 200 \
  --dynamic_mask_size True \
  --size_augmentation True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

