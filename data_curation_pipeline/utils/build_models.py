import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import math
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_internvl(model_name="OpenGVLab/InternVL2_5-26B", cache_dir=None):
    path = model_name
    
    def split_model(model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {
            'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
            'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        # if world_size == 4:
        #     num_layers_per_gpu = [0, 18, 16, 16]
            # num_layers_per_gpu = [2, 20, 20, 8]
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    if "26B" in path:
        device_map = split_model('InternVL2_5-26B')
        # assert torch.cuda.device_count() <= 4, "Do not waste GPUs"
        model = AutoModel.from_pretrained(
            path,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    if "8B" in path:
        device_map = split_model('InternVL2_5-8B')
        model = AutoModel.from_pretrained(
            path,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    elif "38B" in path:
        # assert torch.cuda.device_count() <= 4, "Do not waste GPUs"
        device_map = split_model('InternVL2_5-38B')
        model = AutoModel.from_pretrained(
            path,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
    else:
        model = AutoModel.from_pretrained(
            path,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=True, temperature=0.2)
    
    return model, tokenizer, generation_config

def build_internvl_lmdeploy(model_name="OpenGVLab/InternVL2_5-8B", cache_dir=None, tp=4):
    from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
    from lmdeploy.vl import load_image
    from lmdeploy.vl.constants import IMAGE_TOKEN
    
    backend_config = TurbomindEngineConfig(session_len=16384, tp=tp, cache_max_entry_count=0.8)
    pipe = pipeline(model_name, backend_config=backend_config)
    
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=0.2,
        max_new_tokens=2048)
    
    return pipe, None, gen_config
    
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
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

def load_cv2_image(cv2_image, input_size=448, max_num=12):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_image)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def build_model(model_name, use_lmdeploy=True, cache_dir=None, tp=4):
    if "InternVL" in model_name:
        if use_lmdeploy:
            model, tokenizer, generation_config = build_internvl_lmdeploy(model_name, cache_dir, tp=tp)
        else:
            model, tokenizer, generation_config = build_internvl(model_name, cache_dir)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, tokenizer, generation_config

if __name__ == "__main__":
    build_internvl_lmdeploy("OpenGVLab/InternVL2_5-8B", cache_dir=None)