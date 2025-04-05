from typing import List, Dict, Any
import numpy as np
from pycocotools import mask
import json, os

def calculate_tree_depth(node: Dict[str, Any]) -> int:
    if not node["children"]: 
        return 1
    return 1 + max(calculate_tree_depth(child) for child in node["children"])

def count_total_nodes(tree: Dict[str, Any]) -> int:
    if not tree["children"]:
        return 1
    return 1 + sum(count_total_nodes(child) for child in tree["children"])

def get_deepest_child_of_root(root):
    max_depth = 0
    max_area = -1
    deepest_child_id = None
    
    for child in root["children"]:
        depth = calculate_tree_depth(child)
        if depth > max_depth:
            max_depth = depth
            max_area = child["area"]
            deepest_child_id = child["id"]
        elif depth == max_depth:
            if child["area"] > max_area:
                max_area = child["area"]
                deepest_child_id = child["id"]
    
    return deepest_child_id

def find_subtree_root(tree, node_id):
    if tree["id"] == node_id:
        return tree
    for child in tree["children"]:
        result = find_subtree_root(child, node_id)
        if result:
            return result
    return None

def find(json_data, node_id):
    for ann in json_data["annotations"]:
        if ann["id"] == node_id:
            return ann
    return None

def bbox_overlap(bbox1, bbox2):
    """
    Uses bbox information to check if two bounding boxes overlap
    This is used to accelerate the process of checking if two masks overlap
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2
    
    if x1_max < x2 or x2_max < x1 or y1_max < y2 or y2_max < y1:
        return False
    return True

def intersection_ratio(small_mask, large_mask):
    inter = np.logical_and(small_mask, large_mask).sum()
    small_area = small_mask.sum()
    if small_area == 0:
        return 0.0
    return inter / small_area

def find_deepest_parent(current_node, small_id, annotation_info, threshold):
    if current_node["id"] != -1:
        large_mask_decoded = annotation_info[current_node["id"]]["decoded_mask"]
        small_mask_decoded = annotation_info[small_id]["decoded_mask"]
        ratio = intersection_ratio(small_mask_decoded, large_mask_decoded)
        if ratio < threshold:
            return None  
    
    for child in current_node["children"]:
        deeper_parent = find_deepest_parent(child, small_id, annotation_info, threshold)
        if deeper_parent is not None:
            return deeper_parent
    
    return current_node

def get_children_with_min_depth(root, target_depth=3):
    qualified_children = []
    
    for child in root["children"]:
        depth = calculate_tree_depth(child)
        if depth >= target_depth:
            qualified_children.append(child["id"])
    
    return qualified_children


def build_inclusion_tree(json_data, threshold=0.8):
    """
    Sort json_data["annotations"] in descending order by area,
    and construct a tree by inserting smaller objects into the "deepest" suitable parent.
    
    Tree nodes only store the following fields:
    {
      "id": <annotation_id or -1 for root>,
      "area": <annotation_area>,
      "children": [ ... ]
    }
    """
    annotations = sorted(json_data["annotations"], key=lambda x: x["area"], reverse=True)

    root_node = {
        "id": -1,
        "area": 0,
        "children": []
    }

    annotation_info = {}
    for ann in annotations:
        ann_id = ann["id"]
        annotation_info[ann_id] = {
            "bbox": ann["bbox"],
            "decoded_mask": mask.decode(ann["segmentation"])
        }

    for ann in annotations:
        ann_id = ann["id"]
        ann_area = ann["area"]
        ann_bbox = ann["bbox"]
        ann_segmentation = ann["segmentation"]
        
        new_node = {
            "id": ann_id,
            "area": ann_area,
            "children": [],
            "segmentation": ann_segmentation,
            "bbox": ann_bbox
        }
        
        best_parent = None
        
        def find_any_parent(current):
            if current["id"] != -1:
                if not bbox_overlap(annotation_info[current["id"]]["bbox"], ann_bbox):
                    return None

            return find_deepest_parent(current, ann_id, annotation_info, threshold)

        candidate_parent = find_any_parent(root_node)
        if candidate_parent is not None:
            best_parent = candidate_parent
            
        if best_parent is None:
            root_node["children"].append(new_node)
        else:
            best_parent["children"].append(new_node)

    return root_node

def process_s1_json_file(task):
    """ Single task to process a JSON file and save the resulting tree """
    json_file, tree_dir = task
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    tree = build_inclusion_tree(json_data)

    assert count_total_nodes(tree) == len(json_data["annotations"]) + 1, \
        f"Total nodes mismatch {count_total_nodes(tree)} != {len(json_data['annotations']) + 1}"

    output_path = os.path.join(tree_dir, os.path.basename(json_file))
    with open(output_path, "w") as f:
        json.dump(tree, f, indent=4)

    return f"Processed {json_file}"

if __name__ == "__main__":
    import json
    import os
    import glob
    from tqdm import tqdm
    
    root_dir = "/path/to/root"
    save_dir = "/path/to/save"
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(root_dir, "*.json"))

    for json_file in tqdm(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        try:
            tree = build_inclusion_tree(json_data)
            assert count_total_nodes(tree) == len(json_data["annotations"])+1, f"Total nodes mismatch {count_total_nodes(tree)} != {len(json_data['annotations'])+1}"
            with open(os.path.join(save_dir, os.path.basename(json_file)), "w") as f:
                json.dump(tree, f, indent=4)
        except Exception as e:
            print("Error in", json_file)
            print(e)
            continue
        
        
    
    
    