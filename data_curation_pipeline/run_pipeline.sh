#!/bin/bash

NUM_PROCESSES=4
IMG_DIR="/path/to/images"
JSON_DIR="/path/to/jsons" # Usually IMG_DIR and JSON_DIR are the same
OUTPUT_DIR="/path/to/output"
MLLM_MODEL="OpenGVLab/InternVL2_5-32B-MPO"
TENSOR_PARALLEL=4

# Build mask tree
python run_data_pipeline.py --stage 1 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy

# Find candidates
python run_data_pipeline.py --stage 2 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy

# Prepare images
python run_data_pipeline.py --stage 3 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy

# Generate short captions
python run_data_pipeline.py --stage 4 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy

# Generate long captions
python run_data_pipeline.py --stage 5 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy 

# Generate unique captions
python run_data_pipeline.py --stage 6 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy

# Convert to JSONL format
python run_data_pipeline.py --stage 7 --num_processes ${NUM_PROCESSES} \
    --img_dir ${IMG_DIR} \
    --json_dir ${JSON_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mllm_model ${MLLM_MODEL} \
    --tp ${TENSOR_PARALLEL} \
    --lmdeploy

