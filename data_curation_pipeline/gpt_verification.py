import base64
import openai
import json
import os
from PIL import Image

templates = [
    {
        "role": "system", "content": (
        "You are an expert annotator responsible for validating image captions.\n\n"
            "Given:\n"
            "- A cropped image with contours showing the ROI.\n"
            "- A masked-out version of the ROI.\n\n"
            "Your task:\n"
            "1. Analyze whether the given caption accurately describes the global context of the image and its ROI.\n\n"
            "2. Return **True** if:\n"
            "   - The caption is well-aligned with the images.\n"
            "   - Only minor grammatical or wording issues exist that do not affect meaning.\n"
            "   - *Note:* Do not refine the caption—minor issues are acceptable if meaning remains correct.\n\n"
            "3. Return **False** if:\n"
            "   - The caption contains major errors (e.g., wrong objects, context mismatch, or critical missing details).\n"
            "   - The overall meaning does not align with the given images.\n\n"
            "⚠️ *Important:* The output must be either `True` or `False`. Do not provide explanations or refined captions."
    )},
    {
        "role": "system", "content": (
        "You are an expert annotator responsible for validating image captions.\n\n"
            "Given:\n"
            "- A global image with a highlighted region of interest (ROI).\n"
            "- A cropped image with contours showing the ROI.\n"
            "- A masked-out version of the ROI.\n\n"
            "Your task:\n"
            "1. Analyze whether the given caption accurately describes the global context of the image and its ROI.\n\n"
            "2. Return **True** if:\n"
            "   - The caption is well-aligned with the images.\n"
            "   - Only minor grammatical or wording issues exist that do not affect meaning.\n"
            "   - *Note:* Do not refine the caption—minor issues are acceptable if meaning remains correct.\n\n"
            "3. Return **False** if:\n"
            "   - The caption contains major errors (e.g., wrong objects, context mismatch, or critical missing details).\n"
            "   - The overall meaning does not align with the given images.\n\n"
            "⚠️ *Important:* The output must be either `True` or `False`. Do not provide explanations or refined captions."
    )},
]

def make_caption_query(caption):
    query_dict={
        "type": "text", "text": (
            "### Original Caption:\n"
            f"\"{caption}\"\n\n"
            "### Output Format:\n"
            "- True (if well-aligned)\n"
            "- False (if major errors exist)"
        )}

    return query_dict

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def convert_images_path_to_query(images:list):
    encoded_images = []
    for image_path in images:
        if os.path.exists(image_path):
            Image.open(image_path).convert('RGB').save(os.path.join("/home/jw1510/gpt_verification_real", image_path.split("/")[-1]))
            encoded_images.append(encode_image(image_path))
        else:
            continue

    return encoded_images

def prepare_gpt_query(images, caption):
    messages = []
    if len(images) == 2:
        template = templates[0]
    else: # 3 Images in the image
        template = templates[1]

    messages.append(template)

    # Caption and Images append
    context_dict = {
        "role": "user",
        "content": []

    }
    caption_dict = make_caption_query
    context_dict['content'].append(caption_dict(caption))

    for image in images:
        image_dict = {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{image}"
        }}
        context_dict['content'].append(image_dict)

    messages.append(context_dict)

    return messages

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_path", type=str, default="/home/jw1510/regional_captioning/output_symlinks/outputs_dist/caption_internvl_format.jsonl")
    parser.add_argument("--preprocessed_root", type=str, default="/home/jw1510/regional_captioning/output_symlinks/outputs_dist/preprocessed")
    parser.add_argument("--save_path", type=str, default="/home/jw1510/regional_captioning/output_symlinks/outputs_dist/filtered.jsonl")
    args = parser.parse_args()
    
    testset_path = args.testset_path
    preprocessed_root = args.preprocessed_root
    save_path = args.save_path

    testset = []
    with open(testset_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            testset.append(json.loads(line))

    client = openai.OpenAI()
    filtered_testset = []
    
    filtered_id = 0
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                filtered_testset.append(json.loads(line))
        filtered_id = len(filtered_testset)
                    
    for test_anno in testset:
        global_image_path = os.path.join(preprocessed_root, test_anno["image"] + "_" + test_anno['mask_id'] + ".jpg")
        contour_image_path = os.path.join(preprocessed_root, test_anno["image"] + "_" + test_anno['mask_id'] + "_contour.jpg")
        cropped_image_path = os.path.join(preprocessed_root, test_anno["image"] + "_" + test_anno['mask_id'] + "_cropped.jpg")

        image_path_list = [global_image_path, contour_image_path, cropped_image_path]

        encoded_images = convert_images_path_to_query(image_path_list)
        if len(encoded_images) <= 1:
            continue

        message = prepare_gpt_query(encoded_images, test_anno["conversations"][1]['value'])
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=message,
            temperature=0.0
        )

        gpt_bool = response.choices[0].message.content

        if gpt_bool.lower() == "true":
            new = test_anno.copy()
            new['id'] = filtered_id
            filtered_testset.append(new)
            filtered_id += 1
            with open(save_path, "a") as f:
                f.write(json.dumps(new) + "\n")

        # print(response.choices[0].message.content)
    print(f"Filtered Captions: {len(filtered_testset)}")
    
        
