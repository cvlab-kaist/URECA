# Setups

You must prepare the SA-1B dataset from the following link: [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/)

### Environment Setup

WIP

# Run Data Pipeline

Please set the appropriate variables for `run_pipeline.sh` as described below:

### Argument Descriptions:

- **IMG_DIR**: Path to the directory that contains SA-1B images.
- **JSON_DIR**: Path to the directory that contains the corresponding JSON files for the SA-1B dataset.
- **OUTPUT_DIR**: Path to save the outputs.
- **MLLM_MODEL**: Only works for the family of InternVL-2.5 models (e.g., InternVL-2.5-32B MPO, InternVL-2.5 8B, etc.).
- **TENSOR_PARALLEL**: Valid when using lmdeploy; otherwise, only model parallelism is used.

### Generate Data

To generate the full dataset, please run the following command:

```
bash run_pipeline.sh
```

In our paper, we utilized a 32B model on 4 A100 (40GB) GPUs for generating data.

# Visualize Data

Once you have generated the dataset, you can run `demo.py` to visualize the dataset.

- **save_path**: Path to save captions for visualization format. If it doesn't exist, it will automatically be created based on `caption_path`.
- **caption_path**: Path for the original caption file. **This file must be in the format of `captions_long.json` or `captions_long_refined.json`**, which will be generated in stage 5 or 6 respectively in `run_data_pipeline.py`.
- **img_dir**: Path to the directory that contains SA-1B images.
- **json_dir**: Path to the directory that contains the corresponding JSON files for the SA-1B dataset.

```
python demo.py --save_path /path/to/save/path --caption_path /path/to/caption/path --img_dir /path/to/img/dir --json_dir /path/to/json/dir
```

# GPT Verification

For the test set, we provide an additional verification stage. You can run `gpt_verification.sh` to generate a filtered test set.

- **testset_path**: Path to the generated test set in InternVL JSONL format.
- **preprocessed_root**: Directory for preprocessed images (contours and crops) which is generated in stage 3 in `run_data_pipeline.py`.
- **save_path**: Path to save the filtered test set.

Set your `OPENAI_API_KEY` and run the following bash file:

```
bash gpt_verification.sh
```
