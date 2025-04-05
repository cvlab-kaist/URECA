import json
import os
import shutil

def copy_files(src_folder, dest_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dest_path = os.path.join(dest_folder, filename)

        # Check if it's a file before copying
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied: {filename}")
        else:
            print(f"Skipped (not a file): {filename}")

    print("All files copied successfully!")

def jsonl_files_merge(files_path, testset_path, target_path):
    testset_img_id, testset_mask_id = set(), set()
    with open(testset_path) as f:
        data_list = f.readlines()
        for data in data_list:
            data = json.loads(data)

            testset_img_id.add(data['image'])
            testset_mask_id.add(data['mask_id'])

    final_data = []

    for file_path in files_path:
        with open(file_path) as f:
            data_list = f.readlines()
            for data in data_list:
                final_data.append(json.loads(data))

    with open(target_path, "w") as f:
        i =0
        for data in final_data:
            if data['image'] in testset_img_id or data['mask_id'] in testset_mask_id:
                continue
            data['id'] = i
            f.write(json.dumps(data) + "\n")
            i += 1
    breakpoint()

# source_files = [
#     "/scratch/slurm-user24-kaist/sangbeom/media/data/mug-cap/a100/raw/caption_internvl_format.jsonl",
#     '/scratch/slurm-user24-kaist/sangbeom/media/data/mug-cap/a100/raw/caption_internvl_format_2.jsonl'
# ]
# testset_file = "/scratch/slurm-user24-kaist/sangbeom/media/data/mug-cap/test/raw/caption_internvl_format_eval_2.jsonl"
# target_path = "/scratch/slurm-user24-kaist/sangbeom/media/data/mug-cap/a100/final/caption_internvl_format.jsonl"
#
# jsonl_files_merge(source_files, testset_file, target_path)

# # Example usage
# source_folder = '/scratch/slurm-user24-kaist/junwan/outputs/shard2/sa-1b'
# destination_folder = '/scratch/slurm-user24-kaist/sangbeom/media/data/mug-cap/sa-1b/shard0'
# copy_files(source_folder, destination_folder)