export OPENAI_API_KEY="sk-..."

python gpt_verification.py \
    --testset_path /path/to/testset.jsonl \
    --preprocessed_root /path/to/preprocessed_root \
    --save_path /path/to/save_path.jsonl
