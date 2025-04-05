# Data Preparation (Option)
baseline annotation files does not have `<mask>` tokens, therefore, you should manually add `<mask>` token to your annotation.
```bash
python dataset/mask_token_add_to_jsonl.py
```

# Train Model
First, you need to train mask MLP layer to migrate mask to the pretrained language embedding (InternVL)
```bash
sh shell/mug-cap/cvlab_finetune_stage1.sh
```
Secondly, once you trained mask MLP layer, it is time to train LLM with LoRA.
```bash
sh shell/mug-cap/cvlab_finetune_stage2.sh
```

# Merge LoRA weight
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python tools/merge_lora.py <input_path> <output_path>
```

# Wrap Model
To user `AutoModel` in huggingface, you should make codes for the model to run independtly on the pretrained model folder.
```bash
cp pretrained/InternVL2-2B/*.py work_dirs/internvl_chat_v2_0/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_lora_coco_merge/
```

# Inference
Largely implementated by `https://huggingface.co/OpenGVLab/InternVL2_5-4B`.

```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

# Set if_mask = True if you wish to use dynamic mask processing as well.
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False, if_mask=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    if if_mask:
        resized_img = image.resize((target_width, target_height), Image.NEAREST)
    else:
        resized_img = image.resize((target_width, target_height))

    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)

        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))

        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

import torch
from transformers import AutoTokenizer, AutoModel
path = "Our MUG-CAP Path"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)


if mask_dynamic:
    # mask should be in same resolution with loaded image.
    mask = Image.fromarray(mask.astype(dtype=np.float64) * 255)
    masks = dynamic_preprocess(mask, max_num=12, use_thumbnail=True, if_mask=True)
    mask_values = torch.tensor([np.array(mask).astype(np.uint8) for mask in masks]).unsqueeze(1).to(torch.bfloat16).cuda()
else: # No mask dynamic
    # mask_resized should be binary mask and has shape of [H,W]
    mask_values = torch.tensor(cv2.resize(mask.astype(np.uint8), (448, 448), interpolation=cv2.INTER_NEAREST)).unsqueeze(0).unsqueeze(1).to(
        torch.bfloat16).cuda()

question = '<image>\n<mask>\nDescribe this region in the image.'
response = model.chat(tokenizer, pixel_values, mask_values, question, generation_config)

decoded_text = f'User: {question}\nAssistant: {response}'
```

# Demo
```bash
sh gradio_demo/run.sh
```

# Requirements
Need python version above 3.9.
```bash
pip install -r requirements.txt
```