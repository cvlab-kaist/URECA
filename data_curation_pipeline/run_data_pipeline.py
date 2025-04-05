from utils.build_models import build_model, load_image, load_cv2_image
from utils.prompts import make_prompt
from utils.make_tree import find_subtree_root, process_s1_json_file
from utils.preprocess import (process_single_image, get_candidates, process_crop_and_contour,
                        get_cropped_image_with_masks, get_subset_matrix)
from utils.image_sim_dino import load_dino_model, extract_features, compute_similarity

import torch
import json, os, glob
from tqdm import tqdm
from multiprocessing import Pool
from lmdeploy.vl import load_image as load_image_lmdeploy

def preprocess_1(
        json_dir, output_dir, 
        num_processes=4,
        debug=False,
    ):
    """ Build inclusion tree for each image."""
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        json_files = glob.glob(os.path.join(json_dir, "*.json"))[:20]
    else:
        json_files = glob.glob(os.path.join(json_dir, "*.json"))

    with Pool(processes=num_processes) as pool:
        tasks = [(json_file, output_dir) for json_file in json_files]
        results = list(tqdm(pool.imap(process_s1_json_file, tasks), total=len(tasks)))

    print("Finished processing all JSON files.")

def preprocess_2(
        tree_dir, output_dir,
        num_processes=4,
    ):
    """Find the candidate object of each image."""

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "candidates.json")
    trees = glob.glob(f"{tree_dir}/*.json")

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(get_candidates, trees), total=len(trees)))

    candidates = {}
    for result in results:
        candidates.update(result)

    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=4)

def preprocess_3(
        candidates_dir, tree_dir, img_dir, output_dir,
        color="yellow",
        thickness=3,
        main_object_threshold_max=0.7,
        main_object_threshold_min=0.007,
        sub_object_threshold_min=0.007,
        num_processes=4,
    ):
    """ Get contoured image / cropped image for each submasks.
    Filter out too small or too large masks"""

    os.makedirs(output_dir, exist_ok=True)
    candidates_path = os.path.join(candidates_dir, "candidates.json")
    with open(candidates_path, "r") as f:
        candidates = json.load(f)

    tasks = [
        (img_id, masks, tree_dir, img_dir, output_dir, color, thickness, main_object_threshold_max, main_object_threshold_min, sub_object_threshold_min)
        for img_id, masks in candidates.items()
    ]

    with Pool(processes=num_processes) as pool:
        sub_skipped = list(tqdm(pool.imap(process_crop_and_contour, tasks), total=len(tasks)))

    total_sub_skipped = sum(sub_skipped)
    print(f"Total {total_sub_skipped} submasks skipped.")

def generate_short_captions(
        candidates_dir, preprocessed_dir, tree_dir, output_dir,
        model_name = "OpenGVLab/InternVL2_5-26B",
        use_lmdeploy = True,
        color = "yellow",
        cache_dir = None,
        resume = False,
        tp = 4,
    ):
    """ Generate short captions for each masks. """

    candidates_path = os.path.join(candidates_dir, "candidates.json")
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, generation_config = build_model(model_name, cache_dir=cache_dir, use_lmdeploy=use_lmdeploy, tp=tp)
    with open(candidates_path, "r") as f:
        candidates = json.load(f)
    output_path = os.path.join(output_dir, "captions_short.json")
    captions = {}
    start = 0
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            captions = json.load(f)
            start = len(captions)
    
    for img_id, masks in tqdm(candidates.items()):
        tree_path = os.path.join(tree_dir, f"{img_id}.json")
        with open(tree_path, "r") as f:
            tree = json.load(f)

        for mask_id in masks:
            start_node_id = int(mask_id)
            subtree_root = find_subtree_root(tree, start_node_id)
            if not subtree_root:
                raise ValueError(f"Node with ID {start_node_id} not found in the tree.")

            queue = [(subtree_root, None)]
            while queue: 
                current_node, parent_node = queue.pop(0)
                if img_id in captions.keys() and str(current_node['id']) in captions[img_id].keys():
                    print(f"Skipping {img_id}: {current_node['id']}")
                    continue

                if os.path.exists(os.path.join(preprocessed_dir, f"{img_id}_{current_node['id']}_contour.jpg")):
                    with_parent_path = os.path.join(preprocessed_dir, f"{img_id}_{current_node['id']}_contour.jpg")
                    only_path = os.path.join(preprocessed_dir, f"{img_id}_{current_node['id']}_cropped.jpg")

                    if parent_node:
                        parent_caption = captions[img_id][parent_node["id"]]["short_caption"]
                        sibling_captions = []
                        for sibling in parent_node["children"]:
                            if sibling["id"] in captions[img_id].keys():
                                sibling_captions.append(captions[img_id][sibling["id"]]["short_caption"])
                        if len(sibling_captions) > 0:
                            description2 = ""
                            for idx, sibling_caption in enumerate(sibling_captions):
                                description2 += f"Caption {idx+1}: {sibling_caption}\n"
                            description2 = description2.strip()
                            prompt = make_prompt(stage="generate_short_captions_unique", description=parent_caption, color=color, use_lmdeploy=use_lmdeploy, description2=description2)
                        else:
                            prompt = make_prompt(stage="generate_short_captions", description=parent_caption, color=color, use_lmdeploy=use_lmdeploy)
                    else:
                        prompt = make_prompt(stage="generate_short_captions_main_object", color=color, use_lmdeploy=use_lmdeploy)

                    if use_lmdeploy:
                        pixel_values = [load_image_lmdeploy(with_parent_path), load_image_lmdeploy(only_path)]
                        short_caption = model((prompt, pixel_values), gen_config=generation_config).text
                    else:
                        pixel_values1 = load_image(with_parent_path, max_num=12).to(torch.bfloat16).cuda()
                        pixel_values2 = load_image(only_path, max_num=12).to(torch.bfloat16).cuda()
                        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
                        short_caption = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)

                    if img_id not in captions.keys():
                        captions[img_id] = {}

                    captions[img_id][current_node["id"]] = {
                        "short_caption": short_caption
                    }

                    with open(output_path, "w") as f:
                        json.dump(captions, f, indent=4)
                    
                    
                queue.extend((child, current_node) for child in current_node["children"])
        
        # This is for checkpointing
        if (len(captions) % 1000) == 0 and len(captions) > start:
            with open(os.path.join(output_dir, f"captions_short_{len(captions)}.json"), "w") as f:
                json.dump(captions, f, indent=4)

def generate_long_captions(
        short_caption_dir, tree_dir, output_dir, preprocessed_dir,
        model_name="OpenGVLab/InternVL2_5-26B",
        color="yellow",
        resume=False,
        use_lmdeploy=True,
        cache_dir=None,
        tp=4,
    ):
    """
    Generate long captions by aggregating short captions from children nodes.
    Traverse from bottom to up
    """

    model, tokenizer, generation_config = build_model(model_name, cache_dir=cache_dir, use_lmdeploy=use_lmdeploy, tp=tp)
    short_caption_path = os.path.join(short_caption_dir, "captions_short.json")
    os.makedirs(output_dir, exist_ok=True)
    long_caption_output_path = os.path.join(output_dir, "captions_long.json")

    with open(short_caption_path, "r") as f:
        short_captions = json.load(f)

    long_captions = {}
    if resume and os.path.exists(long_caption_output_path):
        with open(long_caption_output_path, "r") as f:
            long_captions = json.load(f)

    for img_id in tqdm(short_captions.keys(), desc="Processing long captions"):
        if img_id in long_captions.keys():
            continue
        
        tree_path = os.path.join(tree_dir, f"{img_id}.json")
        with open(tree_path, "r") as f:
            tree = json.load(f) 

        filtered_children = {}
        filtered_parent = {}

        queue = [tree]
        while queue:
            node = queue.pop(0)
            node_id = str(node["id"])
            children = node.get("children", [])
            for child in children:
                child_id = str(child["id"])
                queue.append(child)
                if (node_id in short_captions[img_id]) and (child_id in short_captions[img_id]):
                    filtered_parent[child_id] = node_id
                    if node_id not in filtered_children:
                        filtered_children[node_id] = []
                    filtered_children[node_id].append(child_id)

        short_node_ids = set(short_captions[img_id].keys())

        if img_id not in long_captions:
            long_captions[img_id] = {}
        for nid in short_node_ids:
            if nid not in long_captions[img_id]:
                long_captions[img_id][nid] = {
                    "short_caption": short_captions[img_id][nid]["short_caption"],
                    "long_caption": None
                }

        processing_queue = []
        for nid in short_node_ids:
            if nid not in filtered_children or not filtered_children[nid]:
                long_captions[img_id][nid]["long_caption"] = long_captions[img_id][nid]["short_caption"]
                processing_queue.append(nid)

        updated = True
        while updated:
            updated = False
            for child_id, parent_id in list(filtered_parent.items()):
                if parent_id not in short_node_ids:
                    continue
                if long_captions[img_id][parent_id]["long_caption"] is not None:
                    continue
                children_ids = filtered_children.get(parent_id, [])
                all_children_ready = True
                children_long_captions = []
                for cid in children_ids:
                    if long_captions[img_id][cid]["long_caption"] is None:
                        all_children_ready = False
                        break
                    children_long_captions.append(long_captions[img_id][cid]["long_caption"])
                if all_children_ready:
                    parent_short = short_captions[img_id][parent_id]["short_caption"]
                    if children_long_captions:
                        merged_caption = ""
                        for idx, child_caption in enumerate(children_long_captions):
                            merged_caption += f"Subpart {idx + 1} caption: {child_caption}\n"
                        merged_caption = merged_caption.strip()
                        prompt = make_prompt(stage="generate_long_captions", description1=parent_short, description2=merged_caption, color=color, use_lmdeploy=use_lmdeploy)
                        if filtered_parent.get(parent_id) is None:
                            image_path = os.path.join(preprocessed_dir, f"{img_id}_{parent_id}_contour.jpg")
                        else:
                            image_path = os.path.join(preprocessed_dir, f"{img_id}_{parent_id}.jpg")

                        if use_lmdeploy:
                            pixel_values = load_image_lmdeploy(image_path)
                            parent_long = model((prompt, pixel_values), gen_config=generation_config).text
                        else:
                            pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
                            parent_long = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False)
                    else:
                        parent_long = parent_short
                    long_captions[img_id][parent_id]["long_caption"] = parent_long
                    updated = True

                    with open(long_caption_output_path, "w") as f:
                        json.dump(long_captions, f, indent=4)

        for nid in short_node_ids:
            if long_captions[img_id][nid]["long_caption"] is None:
                long_captions[img_id][nid]["long_caption"] = long_captions[img_id][nid]["short_caption"]

    with open(long_caption_output_path, "w") as f:
        json.dump(long_captions, f, indent=4)

    print(f"Long captions saved to {long_caption_output_path}")

def generate_unique_captions(
        captions_dir, preprocessed_dir, img_dir, json_dir,
        model_name="OpenGVLab/InternVL2_5-26B",
        use_lmdeploy=True,
        cache_dir=None,
        tp=4,
        resume=False,
        sim_threshold=0.8,
        add_subset=False,
    ):
    """
    Generate unique captions for each mask using DINO similarity matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dino_model = load_dino_model(device) # Load model to Selected device (GPU/CPU)
    model, tokenizer, generation_config = build_model(model_name, cache_dir=cache_dir, use_lmdeploy=use_lmdeploy, tp=tp)

    caption_path = os.path.join(captions_dir, "captions_long.json")
    with open(caption_path, "r") as f:
        captions = json.load(f)

    output_path = os.path.join(captions_dir, "captions_long_refined.json")
    refined_captions = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            refined_captions = json.load(f)

    for img_id, masks in tqdm(captions.items(), desc="Refine long captions"):

        if img_id in refined_captions.keys(): 
            continue
        else:
            if img_id not in refined_captions:
                refined_captions[img_id] = {}
            for mask_id, caption in masks.items():
                if mask_id not in refined_captions[img_id]:
                    refined_captions[img_id][mask_id] = {}

        cropped_images_path = []

        # Get similarity matrix with DINO
        for mask_id, caption in masks.items():
            if os.path.exists(os.path.join(preprocessed_dir, f"{img_id}_{mask_id}_contour.jpg")):
                cropped_image_path = os.path.join(preprocessed_dir, f"{img_id}_{mask_id}_cropped.jpg")

                cropped_images_path.append(cropped_image_path) 
            else:
                None

        features = extract_features(dino_model, cropped_images_path, device)
        similarity_matrix = compute_similarity(features)

        with open(os.path.join(json_dir, f"{img_id}.json"), "r") as f:
            json_data = json.load(f)

        if add_subset:
            subset_matrix = get_subset_matrix(masks, json_data)

        mask_ids = list(masks.keys())
        for i, (mask_id, caption) in enumerate(masks.items()):
            if add_subset:
                similarity_bool = (similarity_matrix[i] > sim_threshold) & subset_matrix[i]
            else:
                similarity_bool = similarity_matrix[i] > sim_threshold
            similar_mask_ids = [mask_ids[o] for o in range(len(mask_ids)) if similarity_bool[o] and o != i]
            similar_mask_ids.insert(0, mask_ids[i])

            if len(similar_mask_ids) > 1:
                cropped_image = get_cropped_image_with_masks(similar_mask_ids, img_id, img_dir, json_data)

                prompt = make_prompt(stage="generate_unique_captions", description=caption['long_caption'],
                                     use_lmdeploy=use_lmdeploy)

                if use_lmdeploy:
                    pixel_values = load_image_lmdeploy(cropped_image)
                    refined_caption = model((prompt, pixel_values), gen_config=generation_config).text

                else:
                    pixel_values = load_cv2_image(cropped_image, max_num=12).to(torch.bfloat16).cuda()
                    refined_caption = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False)

                refined_captions[img_id][mask_id]['long_caption'] = refined_caption
            else:
                refined_captions[img_id][mask_id]['long_caption'] = caption['long_caption']

        with open(output_path, "w") as f:
            json.dump(refined_captions, f, indent=4)

def convert_to_jsonl(
        long_caption_dir, output_dir,
        prompt_strategy="random",
    ):
    """
    Convert long captions to JSONL format for InternVL.
    """

    long_caption_path = os.path.join(long_caption_dir, "captions_long.json")
    output_path = os.path.join(output_dir, "caption_internvl_format.jsonl")

    prompt_bag = {
        "short": [
            "Describe this region in the image.",
            "Describe this region in the image shortly.",
            "Describe this region in the image in single phrase.",
            "Summarize what is depicted in this area briefly.",
            "Give a short description of this region.",
            "Provide a brief caption for this part.",
            "Summarize this object in one sentence.",
            "Describe this region in one sentence.",
            "Describe this part of the image in one sentence.",
            "Describe this part of the image.",
        ],
        "long": [
            "Describe this region in the image.",
            "Describe the region in the image in detail.",
            "Please describe the region in the image.",
            "Please provide a detailed description of this region of the image.",
            "Give an in-depth explanation of the object and its characteristics.",
            "Describe all notable aspects of this part of the image.",
            "Describe this part of the image in detail.",
            "Describe this part of the image.",
            "Describe this object in the image in detail.",
            "Describe this object in the image.",
        ],
        "deterministic": ["Describe the region in the image"]
    }

    import random
    with open(long_caption_path, "r") as f:
        captions = json.load(f)

        id = 0
        save_jsonl = []
        for img_id, masks in tqdm(captions.items()):
            for mask_id, captions in masks.items():
                if prompt_strategy == "random":
                    if captions['short_caption'].strip() == captions['long_caption'].strip():
                        prompt = random.choice(prompt_bag["short"])
                    else:
                        prompt = random.choice(prompt_bag["long"])
                elif prompt_strategy == "deterministic":
                    prompt = prompt_bag["deterministic"]
                else:
                    raise ValueError("Invalid prompt strategy.")

                current_data = {
                    'id': id,
                    'image': f"{img_id}",
                    'mask_id': f"{mask_id}",
                    'conversations': [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt", "value": f"{captions['long_caption']}"},
                    ]
                }

                save_jsonl.append(current_data)
                id += 1

    with open(output_path, "w") as f:
        for item in save_jsonl:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--json_dir", type=str, default="./sa-1b", help="Directory containing SA-1B JSON files.")
    parser.add_argument("--img_dir", type=str, default="./sa-1b", help="Directory containing SA-1B images.")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory.")
    parser.add_argument("--mllm_model", type=str, default="OpenGVLab/InternVL2_5-32B-MPO", help="Path to multimodal model.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes for preprocessing.")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel for generating captions. Only valid when using LMDeploy.")
    parser.add_argument("--resume", action="store_true", help="Resume generation, if available.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--lmdeploy", action="store_true", help="Use LMDeploy for generating captions.")
    parser.set_defaults(resume=False)
    args = parser.parse_args()
    
    print(args)

    if args.debug:
        try:
            if torch.cuda.device_count() == 1:
                assert os.path.exists(args.mllm_model.replace("26B", "1B"))
                args.mllm_model = args.mllm_model.replace("26B", "1B")
                args.tp = 1
        except:
            print("Debugging with 26B model.")
        import pdb; pdb.set_trace()

    tree_dir = os.path.join(args.output_dir, "trees")
    preprocess_dir = os.path.join(args.output_dir, "preprocessed")
    if args.stage == 1:
        print("Building inclusion tree for each image.")
        preprocess_1(args.json_dir, tree_dir, args.num_processes, args.debug)

    if args.stage == 2:
        print("Finding candidate objects for each image.")
        preprocess_2(tree_dir, args.output_dir, args.num_processes)

    if args.stage == 3:
        print("Getting contoured image / cropped image for each submasks.")
        preprocess_3(args.output_dir, tree_dir, args.img_dir, preprocess_dir, num_processes=args.num_processes)

    if args.stage == 4:
        print("Generating short captions for each masks. (Top-down)")
        generate_short_captions(
            candidates_dir=args.output_dir,
            model_name=args.mllm_model,
            preprocessed_dir=preprocess_dir,
            tree_dir=tree_dir,
            output_dir=args.output_dir,
            resume=args.resume,
            tp=args.tp,
            use_lmdeploy=args.lmdeploy,
        )

    if args.stage == 5:
        print("Generating long captions for each masks. (Bottom-up)")
        generate_long_captions(
            short_caption_dir=args.output_dir,
            model_name=args.mllm_model,
            tree_dir=tree_dir,
            preprocessed_dir=preprocess_dir,
            output_dir=args.output_dir,
            resume=args.resume,
            tp=args.tp,
            use_lmdeploy=args.lmdeploy,
        )

    if args.stage == 6:
        print("Generating unique captions for each masks.")
        generate_unique_captions(
            captions_dir=args.output_dir,
            preprocessed_dir=preprocess_dir,
            img_dir=args.img_dir,
            json_dir=args.json_dir,
            model_name=args.mllm_model,
            use_lmdeploy=args.lmdeploy,
            tp=args.tp,
            resume=args.resume,
        )

    if args.stage == 7:
        print("Converting long captions to JSONL format.")
        convert_to_jsonl(
            long_caption_dir=args.output_dir,
            output_dir=args.output_dir,
            prompt_strategy="random"
        )