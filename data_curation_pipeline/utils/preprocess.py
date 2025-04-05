import cv2
import os
from pycocotools import mask
import numpy as np
import json
from utils.make_tree import find, find_subtree_root, get_children_with_min_depth
from tqdm import tqdm

def get_candidates(tree, target_depth=4):
    """ Get candidates with target depths """
    try:
        save_dict = {}
        with open(tree, "r") as f:
            original_tree = json.load(f)
        
        deep_children = get_children_with_min_depth(original_tree, target_depth)
        key = os.path.basename(tree).split(".")[0]
        save_dict[key] = deep_children
        return save_dict
        
    except Exception as e:
        print(f"Error processing {tree}: {str(e)}")
        return None

def process_single_image(args):
    """Process a single image and save the results"""
    key, value, json_dir, img_dir, save_dir, contour_dir, color, thickness, max_threshold, min_threshold = args
    mask_id = value['id']

    try:
    # Load JSON data for the specific key
        with open(os.path.join(json_dir, f"{key}.json"), "r") as f:
            ann = json.load(f)

        # Find the specific node in the JSON data
        node = find(ann, mask_id)
        if node['area'] / (node['segmentation']['size'][0] * node['segmentation']['size'][1]) > max_threshold:
            return 1  # Skip this image
        if node['area'] / (node['segmentation']['size'][0] * node['segmentation']['size'][1]) < min_threshold:
            return 1  # Skip this image
        depth = value['depth']
        if depth <= 4:
            return 1  # Skip this image

        # File paths
        original_img_name = f"{key}.jpg"
        original_img_path = os.path.join(img_dir, original_img_name)
        blurred_img_name = f"{key}_{mask_id}.jpg"
        blurred_img_path = os.path.join(save_dir, blurred_img_name)
        cropped_img_name = f"{key}_{mask_id}_cropped.jpg"
        cropped_img_path = os.path.join(save_dir, cropped_img_name)
        contour_img_path = os.path.join(contour_dir, f"{key}_{mask_id}.jpg")

        # Load the image
        image = cv2.imread(original_img_path)
        if image is None:
            return 1  # Skip if the image is not found

        # Apply Gaussian Blur
        blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=100)
        binary_mask = mask.decode(node['segmentation'])
        result_image = np.copy(blurred_image)
        result_image[binary_mask > 0] = image[binary_mask > 0]
        
        image_with_contours = image.copy()
        binary_mask = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crop the image
        x, y, w, h = node['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped_image = result_image[y:y+h, x:x+w]

        # Save the images
        cv2.imwrite(blurred_img_path, result_image)
        cv2.imwrite(cropped_img_path, cropped_image)
        
        if color == "red":
            cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), thickness)
        elif color == "yellow":
            cv2.drawContours(image_with_contours, contours, -1, (0, 255, 255), thickness)
        else:
            raise ValueError(f"Invalid color: {color}")
        cv2.imwrite(contour_img_path, image_with_contours)

        return 0  # Successfully processed
    except Exception as e:
        print(f"Error processing {key}: {e}")
        return 1  # Skip this image due to an error

def prepare_crop_and_contour(tree_data, img_id, masks, img_dir, output_dir, color="yellow", thickness=3, main_object_threshold_max=0.7, main_object_threshold_min=0.1, sub_object_threshold_min=0.01):
    main_skipped = 0
    sub_skipped = 0
    original_image_path = None
    for mask_id in masks:
        subtree_root = find_subtree_root(tree_data, mask_id)
        if not subtree_root:
            raise ValueError(f"Node with ID {mask_id} not found in the tree.")

        queue = [(subtree_root, None)]
        if original_image_path is None:
            original_image_path = os.path.join(img_dir, f"{img_id}.jpg")
            image = cv2.imread(original_image_path)
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=100)
        
        while queue:
            current_node, parent_node = queue.pop(0)  # FIFO 방식
            current_image_path = os.path.join(output_dir, f"{img_id}_{current_node['id']}_cropped.jpg")
            current_image_path_with_parent = os.path.join(output_dir, f"{img_id}_{current_node['id']}_contour.jpg")
            contour_image_path = os.path.join(output_dir, f"{img_id}_{current_node['id']}.jpg")
            mx, my, mw, mh = subtree_root['bbox']
            mx, my, mw, mh = int(mx), int(my), int(mw), int(mh)
            binary_mask_parent = mask.decode(subtree_root['segmentation'])
            if not (os.path.exists(current_image_path) and os.path.exists(current_image_path_with_parent)):
                if not parent_node: # Main object
                    ratio = current_node['area'] / (current_node['segmentation']['size'][0] * current_node['segmentation']['size'][1])
                    if ratio > main_object_threshold_max or ratio < main_object_threshold_min:
                        main_skipped += 1
                        # print(f"Skipping main object {current_node['id']} for image {img_id} with ratio {ratio:.5f}")
                        break                        
                if parent_node and os.path.exists(os.path.join(output_dir, f"{img_id}_{parent_node['id']}_cropped.jpg")):
                    if current_node['area'] / (current_node['segmentation']['size'][0] * current_node['segmentation']['size'][1]) < sub_object_threshold_min:
                        sub_skipped += 1
                        continue
                    
                binary_mask = mask.decode(current_node['segmentation'])
                if parent_node:
                    result_image, result_image_with_parent = np.copy(blurred_image), np.copy(blurred_image)
                    result_image[binary_mask > 0] = image[binary_mask > 0]
                    result_image_with_parent[binary_mask_parent > 0] = image[binary_mask_parent > 0]
                    contour_img = image.copy()
                else:
                    result_image, result_image_with_parent = np.copy(blurred_image), np.copy(image)
                    result_image[binary_mask > 0] = image[binary_mask > 0]
                
                for_contour = (binary_mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if color == "red":
                    cv2.drawContours(result_image_with_parent, contours, -1, (0, 0, 255), thickness)
                    if parent_node:
                        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), thickness)
                elif color == "yellow":
                    cv2.drawContours(result_image_with_parent, contours, -1, (0, 255, 255), thickness)
                    if parent_node:
                        cv2.drawContours(contour_img, contours, -1, (0, 255, 255), thickness)
                else:
                    raise ValueError(f"Invalid color: {color}")
                
                if parent_node:
                    result_image_with_parent = result_image_with_parent[my:my+mh, mx:mx+mw]
                    cv2.imwrite(contour_image_path, contour_img)
                    
                x, y, w, h = current_node['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                result_image = result_image[y:y+h, x:x+w]
                
                cv2.imwrite(current_image_path, result_image)
                cv2.imwrite(current_image_path_with_parent, result_image_with_parent)
            queue.extend((child, current_node) for child in current_node["children"])
    # print(f"Main skipped: {main_skipped}, Sub skipped: {sub_skipped}")
    return sub_skipped

def process_single_image_sub(args):
    """Process a single image for sub-object preparation"""
    tree_dir, img_dir, blur_dir, contour_dir, key, color, thickness, min_threshold = args

    # Parse image details
    _, num, id = key.split("_")
    id = int(id.replace(".jpg", ""))
    
    # Load tree data
    tree_path = os.path.join(tree_dir, f"sa_{num}.json")
    with open(tree_path, "r") as f:
        tree_data = json.load(f)

    # Prepare sub-objects
    img_name = f"sa_{num}"
    skipped = prepare_image_sub(tree_data, img_dir, img_name, blur_dir, contour_dir, id, color, thickness, min_threshold)
    return skipped

def process_crop_and_contour(args):
    """Process a single image for sub-object preparation"""
    img_id, masks, tree_dir, img_dir, output_dir, color, thickness, main_object_threshold_max, main_object_threshold_min, sub_object_threshold_min = args
    tree_path = os.path.join(tree_dir, f"{img_id}.json")
    
    # Load tree data
    tree_path = os.path.join(tree_dir, f"{img_id}.json")
    with open(tree_path, "r") as f:
        tree_data = json.load(f)

    sub_skipped = prepare_crop_and_contour(tree_data, img_id, masks, img_dir, output_dir, color, thickness, main_object_threshold_max, main_object_threshold_min, sub_object_threshold_min)
    return sub_skipped

COLORS = [
    ((255, 0, 0), "Blue"),           # Blue
    ((0, 255, 0), "Green"),          # Green
    ((0, 0, 255), "Red"),            # Red
    ((255, 255, 0), "Cyan"),         # Cyan
    ((255, 0, 255), "Magenta"),      # Magenta
    ((0, 255, 255), "Yellow"),       # Yellow
    ((128, 0, 128), "Purple"),       # Purple
    ((255, 165, 0), "Orange"),       # Orange
    ((0, 128, 128), "Teal"),         # Teal
    ((128, 128, 128), "Gray"),       # Gray
    ((75, 0, 130), "Indigo"),        # Indigo
    ((255, 105, 180), "Pink"),       # Pink
    ((139, 69, 19), "Brown"),        # Brown
    ((46, 139, 87), "Sea Green"),    # Sea Green
    ((173, 216, 230), "Light Blue"), # Light Blue
    ((220, 20, 60), "Crimson"),      # Crimson
    ((244, 164, 96), "Sandy Brown"), # Sandy Brown
    ((154, 205, 50), "Yellow Green"),# Yellow Green
    ((72, 61, 139), "Dark Slate Blue"), # Dark Slate Blue
    ((0, 191, 255), "Deep Sky Blue") # Deep Sky Blue
]
font = cv2.FONT_HERSHEY_SIMPLEX

def draw_contour_with_coco_rle(segmentation, image, index, font_scale=1, font_thickness=2):
    """
    Draws contours from an RLE-encoded mask on the image and adds an index label with a black background.

    Args:
        segmentation: RLE-encoded segmentation mask.
        image: The image on which to draw contours.
        index: Number label to display near the contour.
        font_scale: Scale of the text.
        font_thickness: Thickness of the text.

    Returns:
        The modified image with contours and labels.
    """
    # Get color from list
    color, color_name = COLORS[index % len(COLORS)]  # Ensure it wraps around

    # Decode RLE mask
    decoded_mask = mask.decode(segmentation)  # Shape: [height, width]
    binary_mask = (decoded_mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours with the selected color
    cv2.drawContours(image, contours, -1, color, 2)

    # Find the center of the largest contour for placing text
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = largest_contour[0][0]  # Fallback to first contour point

        # Get text size
        text = str(index)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_w, text_h = text_size

        # Define text background box (tight fit)
        bg_x1, bg_y1 = cX, cY - text_h - 5
        bg_x2, bg_y2 = cX + text_w + 5, cY + 5

        # Draw black background for text
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        # Draw text in the selected color
        cv2.putText(image, text, (cX, cY), font, font_scale, color, font_thickness)

    return image, color_name  # Return the image and the color name for reference


def get_cropped_image_with_masks(similar_mask_ids: list, img_id, img_dir, json_data):
    image = cv2.imread(os.path.join(img_dir, f"{img_id}.jpg"))

    min_x, mim_y, max_x, max_y = None, None, None, None

    reversed_similar_mask_ids = similar_mask_ids[::-1]
    for similar_mask_id in reversed_similar_mask_ids:
        annotation = [annotation for annotation in json_data['annotations'] if str(annotation['id']) == similar_mask_id][0]

        index = similar_mask_ids.index(str(annotation['id']))
        segmentation = annotation['segmentation']
        image, contour_color = draw_contour_with_coco_rle(segmentation, image, index)

        x,y,w,h = annotation['bbox']
        box_min_x, box_min_y, box_max_x, box_max_y = x,y, x+w, y+h

        if min_x is None or min_x > box_min_x:
            min_x = box_min_x
        if mim_y is None or mim_y > box_min_y:
            mim_y = box_min_y
        if max_x is None or max_x < box_max_x:
            max_x = box_max_x
        if max_y is None or max_y < box_max_y:
             max_y = box_max_y

    cropped_image = image[int(mim_y):int(max_y), int(min_x):int(max_x)]

    return cropped_image

def compute_subset_matrix(masks: np.ndarray) -> np.ndarray:
    """
    Compute a boolean matrix indicating whether each mask is a subset of another.

    Args:
        masks (np.ndarray): A numpy array of shape (N, H, W) containing N binary masks.

    Returns:
        np.ndarray: A boolean matrix of shape (N, N) where entry (i, j) is True if mask_i is a subset of mask_j.
    """
    N = masks.shape[0]
    subset_matrix = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(N):
            if i != j:  # Exclude self-check if needed
                subset_matrix[i, j] = np.all((masks[i] == 1) <= (masks[j] == 1))

    return subset_matrix

def get_subset_matrix(masks, json_data):
    decoded_masks = []
    for annotation in json_data['annotations']:
        if str(annotation['id']) in masks.keys():
            segmentation = annotation['segmentation']
            decoded_mask = mask.decode(segmentation)  # Shape: [height, width]
            decoded_masks.append(decoded_mask)
    not_subset_matrix = compute_subset_matrix(np.array(decoded_masks))

    return not_subset_matrix
        
if __name__ == "__main__":
    import json
    import glob
    from tqdm import tqdm
    import pdb
    import shutil
    pdb.set_trace()

    tree_test = "/path/to/tree.json"
    new_dict = get_candidates(tree_test, target_depth=3)
    print(new_dict)
    tree_dir = "/path/to/tree"
    img_dir = "/path/to/image"
    output_dir = "/path/to/output"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    
    args = [
        (img_id, masks, tree_dir, img_dir, output_dir, "yellow", 3, 0.7, 0.05, 0.01)
        for img_id, masks in new_dict.items()
    ]
    
    process_crop_and_contour(args[0])
    
    
    