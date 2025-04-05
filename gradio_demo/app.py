import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import streamlit as st
import gradio as gr
import torch
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import requests
import torch
from transformers import AutoTokenizer, AutoModel
import os


# Load SAM model
device = "cuda"


sam_checkpoint = "./models/sam/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

sam_predictor = SamPredictor(sam)

path = "./models/URECA"
dynamic_mask = True


vlm_model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

"""
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)

question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')
"""

# vlm_model = None
# tokenizer = None


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
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image_file, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def generate_mask(image, input_points_text, input_labels_text, dynamic_mask=True):
    """Processes the user clicks to generate a mask using SAM."""
    if image is None:
        return None, "Please upload an image."

    # Convert input text (comma-separated points) to coordinates
    input_points = np.array([list(map(int, point.split(","))) for point in input_points_text.split(";")])
    input_labels = np.array([int(label) for label in input_labels_text.split(";")])  # Convert label text to integers

    image = np.array(image)
    sam_predictor.set_image(image)

    masks, _, _ = sam_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)

    # Overlay mask on image
    overlay = image.copy()
    overlay[masks[0] > 0] = [0, 255, 0]  # Green overlay for the mask

    # Resize mask to 448x448
    if dynamic_mask:
        mask = Image.fromarray(masks[0].astype(dtype=np.float64) * 255)  # * 255, if value is in [0,1]
        masks = dynamic_preprocess(mask, max_num=12, use_thumbnail=True, if_mask=True)
        mask_values = [np.array(mask).astype(np.uint8) for mask in masks]
    else:
        mask_values = np.expand_dims(cv2.resize(masks[0].astype(np.uint8), (448, 448), interpolation=cv2.INTER_NEAREST), axis=0)

    # Draw the clicked points as red or blue dots based on label
    for (click_x, click_y), label in zip(input_points, input_labels):
        color = (255, 0, 0) if label == 1 else (0, 0, 255)  # Red for foreground, Blue for background
        cv2.circle(overlay, (click_x, click_y), 5, color, -1)

    return overlay, mask_values


def generate_caption(image, mask_resized):
    if image is None or mask_resized is None:
        return "Please generate a mask first."

    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    mask_values = torch.tensor(mask_resized).unsqueeze(1).to(torch.bfloat16).cuda()

    question = '<image>\n<mask>\nDescribe this region in the image.'
    response = vlm_model.chat(tokenizer, pixel_values, mask_values, question, generation_config)

    decoded_text = f'User: {question}\nAssistant: {response}'

    return decoded_text


def click_event(image, input_points_text, input_labels_text, evt: gr.SelectData, foreground_or_background):
    x, y = evt.index[0], evt.index[1]
    label = 1 if foreground_or_background == "Foreground" else 0  # 1 for foreground, 0 for background
    new_input_points_text = f"{input_points_text};{x},{y}" if input_points_text else f"{x},{y}"
    new_input_labels_text = f"{input_labels_text};{label}" if input_labels_text else f"{label}"


    return new_input_points_text, new_input_labels_text

def clear_mask():
    return "", "", None, None  # Reset points, labels, and mask

img_url = "https://sangbeomlim.github.io/images/sangbeom_profile.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw)

def load_selected_image(image_url):
    image = Image.open(requests.get(image_url, stream=True).raw)
    return image


# Pre-selected images (replace with your own if needed)
root = "/home/cvlab14/project/sangbeom/URECA/gradio_demo/gallery"
garm_list_path = [os.path.join(root, path) for path in os.listdir(root)]

with gr.Blocks() as demo:
    gr.Markdown("# URECA Demo")

    # with gr.Row():
    #     for img_url in preselected_images:
    #         gr.Image(value=load_selected_image(img_url), interactive=True, label="Click to Select").click(
    #             load_selected_image, inputs=[gr.State(img_url)], outputs=[gr.Image.update()])

    with gr.Row():
        image_input = gr.Image(value=raw_image, label="Input Image", interactive=True, type="pil")
        output_image = gr.Image()

    with gr.Row():
        example = gr.Examples(
            inputs=image_input,
            examples_per_page=12,
            examples=garm_list_path
        )

    input_points_text = gr.State("")
    input_labels_text = gr.State("")
    mask_resized = gr.State()
    dynamic_mask = gr.State(dynamic_mask)

    with gr.Row():
        foreground_or_background = gr.Radio(["Foreground", "Background"], label="Select Point Type", value="Foreground")
        btn_generate_mask = gr.Button("Generate Mask")
        btn_generate_caption = gr.Button("Generate Caption")
        btn_clear_mask = gr.Button("Clear Mask")

    caption_output = gr.Textbox()

    # New textbox to display clicked coordinates and labels
    input_points_text = gr.Textbox(label="Clicked Coordinate", interactive=True)
    input_labels_text = gr.Textbox(label="Clicked Labels", interactive=True)

    image_input.select(click_event, inputs=[image_input, input_points_text, input_labels_text, foreground_or_background],
                       outputs=[input_points_text, input_labels_text])

    btn_generate_mask.click(generate_mask, inputs=[image_input, input_points_text, input_labels_text, dynamic_mask],
                            outputs=[output_image, mask_resized])
    btn_generate_caption.click(generate_caption, inputs=[image_input, mask_resized], outputs=[caption_output])
    btn_clear_mask.click(clear_mask, inputs=[],
                         outputs=[input_points_text, input_labels_text, output_image, mask_resized])

demo.launch(server_port=7860, debug=True, share=True)
