import gradio as gr
import json
import os
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import argparse
# Paths

parser = argparse.ArgumentParser(description='Gradio App for Regional Captioning')
parser.add_argument('--save_path', type=str, default="./outputs/integrated.json", help='Path to the caption file')
parser.add_argument('--caption_path', type=str, default="./outputs/long_captions.json", help='Path to the caption file')
parser.add_argument('--img_dir', type=str, default="./sa-1b", help='Path to the image directory')
parser.add_argument('--json_dir', type=str, default="./sa-1b", help='Path to the json directory')
args = parser.parse_args()

caption_file = args.save_path
from_dir = args.caption_path
img_dir = args.img_dir
json_dir = args.json_dir

def format_json(caption_file):
    display = {}
    with open(from_dir, "r") as f:
        scene = json.load(f)
    
    for img_id, value in scene.items():
        if img_id not in display.keys():
            display[img_id] = {}
        for mask_id, mask_value in value.items():
            display[img_id][mask_id] = mask_value["long_caption"]
            
    with open(caption_file, "w") as f:
        print("saving at", caption_file)
        json.dump(display, f, indent=4)
    
    return display

if os.path.exists(caption_file):
    with open(caption_file, "r", encoding="utf-8") as f:
        captions_data = json.load(f)
else:
    captions_data = format_json(caption_file)

# Get list of image_ids from caption data
image_ids = list(captions_data.keys())

# Decode segmentation and prepare masks
def decode_segmentation(segmentation):
    return coco_mask.decode(segmentation)

def prepare_masks(filtered_annotations):
    masks = []
    for annotation in filtered_annotations:
        segmentation = annotation["segmentation"]
        mask_id = annotation["id"]
        decoded_mask = decode_segmentation(segmentation)  # Decode binary mask
        masks.append((decoded_mask, f"{mask_id}"))  # Add mask and its label
    return masks

# Load image and annotations dynamically
def load_image_and_masks(image_id):
    # Get image filename and annotation file
    image_filename = f"{image_id}.jpg"
    annotation_file = os.path.join(json_dir, f"{image_id}.json")
    image_path = os.path.join(img_dir, image_filename)
    
    # Load annotations
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)
    annotations = annotations_data["annotations"]
    
    # Get relevant mask IDs
    relevant_mask_ids = set(map(int, captions_data.get(image_id, {}).keys()))  # Mask IDs in captions file
    
    # Filter annotations to only include relevant masks
    filtered_annotations = [annotation for annotation in annotations if annotation["id"] in relevant_mask_ids]
    
    # Prepare masks
    masks = prepare_masks(filtered_annotations)
    
    return image_path, masks

# Gradio event function to display captions
def display_caption(evt: gr.SelectData, masks, image_id):
    # Extract the mask ID from the label
    mask_id = int(masks[evt.index][1])  # Get the label corresponding to the selected mask
    caption_data = captions_data.get(image_id, {}).get(str(mask_id), "No caption found")
    return caption_data

# Gradio event function to update image
def update_image(image_index):
    image_id = image_ids[image_index]
    image_path, masks = load_image_and_masks(image_id)
    # Return the correct tuple structure
    return (image_path, [(m[0], m[1]) for m in masks]), masks, image_id, image_index

# Initialize first image and masks
initial_image_id = image_ids[0]
initial_image_path, initial_masks = load_image_and_masks(initial_image_id)

# Gradio event function to reload JSON data
def reload_data(image_index):
    global captions_data, image_ids
    # Reload the captions data
    captions_data = format_json(caption_file)
    
    # Update image IDs
    image_ids = list(captions_data.keys())
    # Ensure the current index is within the updated range
    image_index = min(image_index, len(image_ids) - 1)
    # Reload the image and masks for the current index
    image_id = image_ids[image_index]
    image_path, masks = load_image_and_masks(image_id)
    return (image_path, [(m[0], m[1]) for m in masks]), masks, image_id, image_index, len(image_ids) - 1

# Add Reload button to the interface
with gr.Blocks() as demo:
    gr.Markdown("## Annotated Image with Filtered Mask Selection and Captions")

    # Annotated Image component
    with gr.Row():
        annotated_img = gr.AnnotatedImage(
            value=(initial_image_path, [(m[0], m[1]) for m in initial_masks]),
            label="Annotated Image",
            height=400,
        )

    # Caption display
    with gr.Row():
        scene_caption = gr.Textbox(label="Generated Caption", interactive=False, lines=5)
        # object_caption = gr.Textbox(label="Object Caption", interactive=False, lines=5)

    # Navigation controls
    with gr.Row():
        prev_button = gr.Button("Prev Image")
        slider = gr.Slider(0, len(image_ids) - 1, step=1, value=0, label="Jump to Image")
        next_button = gr.Button("Next Image")
    with gr.Row():
        reload_button = gr.Button("🔄 Reload Data")

    # State to store current masks and image_id
    current_masks = gr.State(initial_masks)
    current_image_id = gr.State(initial_image_id)

    # Event listener for mask selection
    annotated_img.select(
        fn=display_caption,
        inputs=[current_masks, current_image_id],
        outputs=[scene_caption]
    )

    # Update image based on slider or button
    def prev_image(image_index):
        new_index = max(0, image_index - 1)
        return update_image(new_index)

    def next_image(image_index):
        new_index = min(len(image_ids) - 1, image_index + 1)
        return update_image(new_index)

    prev_button.click(
        fn=prev_image,
        inputs=slider,
        outputs=[annotated_img, current_masks, current_image_id, slider]
    )

    next_button.click(
        fn=next_image,
        inputs=slider,
        outputs=[annotated_img, current_masks, current_image_id, slider]
    )

    slider.release(
        fn=update_image,
        inputs=slider,
        outputs=[annotated_img, current_masks, current_image_id, slider]
    )

    # Reload button functionality
    reload_button.click(
        fn=reload_data,
        inputs=slider,  # Pass the current image index
        outputs=[annotated_img, current_masks, current_image_id, slider, slider]  # Update slider range as well
    )

# Launch the Gradio app
demo.launch(share=True)