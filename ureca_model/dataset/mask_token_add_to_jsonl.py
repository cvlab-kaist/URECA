import json

source = "/home/cvlab14/project/sangbeom/data3/seg_cap/dataset/mug-cap/annotations/test/raw/caption_internvl_format_eval_clean_final.jsonl"
target = "/home/cvlab14/project/sangbeom/data3/seg_cap/dataset/mug-cap/annotations/test/preprocessed/caption_internvl_format_eval_clean_final.jsonl"

position = "after_image"

data_list = []
with open(source, "r") as f:
    for line in f:
        data_list.append( json.loads(line) )

for data in data_list:
    for conversation in data['conversations']:
        if '<image>' in conversation['value'] and position == "after_image":
            conversation['value'] = conversation['value'].replace('<image>', "<image>\n<mask>", 1)

with open(target, "w") as f:
    for data in data_list:
        f.write( json.dumps(data, ensure_ascii=False) + "\n" )

breakpoint()