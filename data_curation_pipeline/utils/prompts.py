from typing import List, Dict, Any, Literal
from lmdeploy.vl.constants import IMAGE_TOKEN


def make_prompt(stage: Literal[
    "generate_short_captions",
    "generate_short_captions_unique",
    "generage_short_captions_main_object",
    "generate_long_captions",
    "generate_unique_captions",
    "internvl_scene",
    "internvl_sub_object",
    "internvl_sub_object_reversed",
    "qwen_object_filter",
    "qwen_main_object_merge",
    "qwen_sub_object_merge",
], use_lmdeploy=True, **kwargs) -> str:
    if stage == "generate_long_captions":
        assert "description1" in kwargs and "description2" in kwargs
        assert 'color' in kwargs
        prompt = f"""{IMAGE_TOKEN if use_lmdeploy else "<image>"}
<task>
    You are a detailed caption generator tasked with describing the main object in images. Your goal is to create precise and detailed captions while avoiding hallucination.
</task>
<objectives>
    1. The caption must primarily focus on the main object while considering its contextual information to clearly identify what it is.
    2. The caption must emphasize the main object’s attributes, such as color, texture, shape, and action if visible.
    3. Describe only what is visible in the image. Avoid adding any information that is not present.
    4. The main object is highlighted with a {kwargs['color']} contour.
    5. A short description of the main object will be provided in the prompt, which can be used to describe the main object.
    6. The main object consists of multiple subparts, and descriptions of these subparts will be provided in the prompt.
    7. The description of subparts may contain inaccurate, unimportant, or redundant information. Use only the essential details that do not contradict the given image to ensure that the caption for the main object compositionally reflects relevant information from these subparts.
</objectives>
<inputDetails>
    1. An image with the main object marked by a {kwargs['color']} contour will be provided.
    2. A short description of the main object will be included in the prompt.
    3. Descriptions of the subparts of the main object will also be provided in the prompt.
</inputDetails>
<descriptionOfMainObject>
    {kwargs["description1"]}
</descriptionOfMainObject>
<descriptionOfSubparts>
    {kwargs["description2"]}    
</descriptionOfSubparts>
<outputFormat>
    1. Provide a single descriptive paragraph that focuses on the main object.
    2. Do not use bullet points or lists.
    3. Incorporate details from the provided descriptions to accurately depict the main object.
    4. Never mention the presence of the {kwargs['color']} contour in any form.
    5. Structure the caption clearly and concisely, avoiding excessive detail or verbosity. Do not start with phrases like “The image shows…”.
    6. Ensure the focus is evident without explicitly stating that it is the main object.
</outputFormat>"""
    elif stage == "generate_short_captions_unique":
        assert "description" in kwargs
        assert "description2" in kwargs
        assert 'color' in kwargs
        prompt = f"""Image-1: {IMAGE_TOKEN if use_lmdeploy else "<image>"}
Image-2: {IMAGE_TOKEN if use_lmdeploy else "<image>"}
<task>
    You are a detailed caption generator tasked with describing the main object in images. Your goal is to create a simple phrase that accurately represents the main object while avoiding hallucination.
</task>
<objectives>
    1. The main object is a subpart of a larger object; therefore, the main object alone may provide insufficient information.
    2. The primary focus of the caption must be on the main object while also considering its positional relationship or functional connection with the larger object.
    3. The primary focus of the caption must be on the main object, emphasizing attributes like color, texture, shape, and action if visible.
    4. The background is blurred to emphasize the main object. Focus solely on describing the main object in detail without mentioning the blurred background.
    5. The caption should be distinguishable from other subparts of the same larger object so that the region can be identified solely by looking at the caption. Therefore, the caption should incorporate positions or attributes that are unique to the main object.
    6. Creating a unique caption is important, but the most critical aspect is accuracy. Do not add unnecessary information solely for the sake of uniqueness.
</objectives>
<inputDetails>
    1. Image-1 highlights the main object with a {kwargs['color']} contour to illustrate its relationship with the larger object.
    2. Image-2 shows the main object cropped from the larger object.
    3. A description of the larger object will be provided in the prompt to help identify the main object.
    4. Descriptions of other subparts of the same larger object will also be provided. The caption for the main object must be clearly distinguishable from the descriptions of these subparts.
</inputDetails>
<descriptionOfLargerObject>
    {kwargs["description"]}
</descriptionOfLargerObject>
<descriptionOfSubparts>
    {kwargs["description2"]}
</descriptionOfSubparts>
<outputFormat>
    1. Provide a simple phrase focusing on the main object while considering its positional relationship or functional connection with the larger object.
    2. The larger object may contain another object with similar attributes to the main object. The caption should be written in a way that clearly distinguishes the main object from these similar objects.
    3. Keep the caption concise, limiting it to one sentence while ensuring clarity and coherence.
    4. Do not explicitly mention the {kwargs['color']} contour or its presence in the image.
    5. Use contextual information from Image-1 to describe the main object’s relationship with the larger object, while referencing its attributes from Image-2.
    6. Contextual details from Image-1 and the description of the larger object should be used only to support the description of the main object.
</outputFormat>
<outputExamples>
    1. The left checkered pocket on the green shirt worn by the man in jeans.
	2. The largest red-sliced strawberry in the salad bowl.
	3. The third window from the right on the second floor of the four-story antique building.
	4. Page 42 of the open book on the wooden table, displaying the inscription “In theory, there’s no difference between theory and practice. In practice there is.”
	5. The metal handle on the white desk shelf.
	6. The four wheels of the purple Maserati parked near the center line.
	7. The number 7 on the red-jersey basketball player’s uniform.
	8. The group of books on the second shelf from the top of the wooden bookcase.
</outputExamples>
"""
    elif stage == "generate_short_captions_main_object":
        assert 'color' in kwargs
        prompt = f"""Image-1: {IMAGE_TOKEN if use_lmdeploy else "<image>"}
Image-2: {IMAGE_TOKEN if use_lmdeploy else "<image>"}
<task>
    You are a detailed caption generator tasked with describing the main object in images. Your goal is to create a simple and accurate phrase for the main object while avoiding hallucination.
</task>
<objectives>
    1. The caption must primarily focus on the main object while considering its positional relationship or functional connection within the entire scene.
    2. The caption must emphasize the main object’s attributes, such as color, texture, shape, and action if visible.
    3. The background of Image-1 is blurred to highlight the main object. Describe the main object in detail without mentioning the blurred background.
</objectives>
<inputDetails>
    1. Image-1 highlights the main object with a {kwargs['color']} contour to illustrate its relationship with the entire scene.
    2. Image-2 displays the main object cropped from the entire scene.
</inputDetails>
<outputFormat>
    1. Provide a simple phrase focusing on the main object while considering its positional relationship or functional connection within the entire scene.
    2. The caption should uniquely identify the main object within the scene.
    3. Keep the caption concise, limiting it to one sentence while ensuring clarity and coherence.
    4. Do not explicitly mention the {kwargs['color']} contour or its presence in the image.
    5. Reference contextual information from Image-1 to describe the main object’s relationship with the scene, while using Image-2 to describe its attributes.
    6. Use contextual information from Image-1 only to support the description of the main object.
</outputFormat>
<outputExamples>  
    1. A young man in a black suit walking along a busy New York street.  
    2. An antique wooden chair with intricate carvings positioned in the corner of the room.  
    3. Page 42 of the open book on the wooden table, displaying the text: "In theory, there’s no difference between theory and practice. In practice, there is."  
    4. A red and white striped umbrella standing upright on the sandy beach.  
    5. A wooden table in the center of the room with a vase of fresh flowers on top.  
    6. A red sports car parked directly in front of a modern glass building.  
    7. A group of lions resting under the shade of a large green tree in the savannah.  
    8. A large blackboard covered with a mathematical equation, including "E=mc^2."  
</outputExamples> 
"""
    elif stage == "generate_short_captions":
        assert "description" in kwargs
        assert 'color' in kwargs
        prompt = f"""Image-1: {IMAGE_TOKEN if use_lmdeploy else "<image>"}
Image-2: {IMAGE_TOKEN if use_lmdeploy else "<image>"}
<task>
    You are a detailed caption generator tasked with describing the main object in images. Your goal is to create a simple phrase that accurately represents the main object while avoiding hallucination.
</task>
<objectives>
    1. The main object is a subpart of a larger object; therefore, the main object alone may provide insufficient information.
    2. The primary focus of the caption must be on the main object while also considering its positional relationship or functional connection with the larger object.
    3. The primary focus of the caption must be on the main object, emphasizing attributes like color, texture, shape, and action if visible.
    4. The background is blurred to emphasize the main object. Focus solely on describing the main object in detail without mentioning the blurred background.
    5. The caption should be distinguishable from other subparts of the same larger object so that the region can be identified solely by looking at the caption. Therefore, the caption should incorporate positions or attributes that are unique to the main object.
    6. Creating a unique caption is important, but the most critical aspect is accuracy. Do not add unnecessary information solely for the sake of uniqueness.
</objectives>
<inputDetails>
    1. Image-1 highlights the main object with a {kwargs['color']} contour to illustrate its relationship with the larger object.
    2. Image-2 shows the main object cropped from the larger object.
    3. A description of the larger object will be provided in the prompt to help identify the main object.
</inputDetails>
<descriptionOfLargerObject>
    {kwargs["description"]}
</descriptionOfLargerObject>
<outputFormat>
    1. Provide a simple phrase focusing on the main object while considering its positional relationship or functional connection with the larger object.
    2. The larger object may contain another object with similar attributes to the main object. The caption should be written in a way that clearly distinguishes the main object from these similar objects.
    3. Keep the caption concise, limiting it to one sentence while ensuring clarity and coherence.
    4. Do not explicitly mention the {kwargs['color']} contour or its presence in the image.
    5. Use contextual information from Image-1 to describe the main object’s relationship with the larger object, while referencing its attributes from Image-2.
    6. Contextual details from Image-1 and the description of the larger object should be used only to support the description of the main object.
</outputFormat>
<outputExamples>
    1. The left checkered pocket on the green shirt worn by the man in jeans.
	2. The largest red-sliced strawberry in the salad bowl.
	3. The third window from the right on the second floor of the four-story antique building.
	4. Page 42 of the open book on the wooden table, displaying the inscription “In theory, there’s no difference between theory and practice. In practice there is.”
	5. The metal handle on the white desk shelf.
	6. The four wheels of the purple Maserati parked near the center line.
	7. The number 7 on the red-jersey basketball player’s uniform.
	8. The group of books on the second shelf from the top of the wooden bookcase.
</outputExamples>
"""
    elif stage == "generate_unique_captions":
        assert 'description' in kwargs
        prompt = f"""{IMAGE_TOKEN if use_lmdeploy else "<image>"}
    <task>
    You are a caption refinement model that enhances given descriptions to generate unique and precise captions for objects in an image. Your goal is to refine the provided caption based on contour-based indexing while maintaining clarity and specificity.
    </task>
    <objectives>
        1. Describe only what is visible in the image. Avoid adding any information that is not present.
        2. The image contains multiple contours in different colors, each with a corresponding index, marking distinct objects.
        3. The main object corresponds to index 0 and is specifically outlined with a blue contour.
        4. Your task is to refine the caption for index 0, highlighting its unique attributes while clearly differentiating it from other indexed contours in the image.
        5. The refined caption must primarily focus on index 0 while considering its contextual information to clearly identify it from other indices.
        6. The caption must emphasize index 0’s attributes, such as color, texture, shape, and action, to make caption unique.
    </objectives>
    <inputDetails>
        1. The contours in the image are color-coded, and each contour has a corresponding index.
        2. The index corresponding to each contour is placed at the center of the contour, matching its color.
        3. The initial caption for index 0 (blue contour) is provided as input.
        4. The refined caption should ensure the distinction between index 0 (blue contour) and other objects in the image.
    </inputDetails>
    <refinementGuidelines>
        1. Preserve the core meaning of the given caption while improving its specificity and uniqueness.
        2. Emphasize key attributes that differentiate index 0 (blue contour) from other indices.
        3. Avoid mentioning the presence of contours or annotations explicitly in the caption.
        4. Keep the refined caption clearly yet descriptive.
        5. Ensure that the final caption remains a natural, human-like description of the object.
        6. Do not use bullet points or lists.
        7. Do not start the answer with words like “Certainly!”.
    </refinementGuidelines>
    <captionForIndex0>
        {kwargs['description']}
    </captionForIndex0>
    <outputFormat>
        1. Provide a single descriptive paragraph that maintains clarity and coherence focusing on index 0 (blue contour)
        2. The refined caption should distinguish index 0 (blue contour) from other indices.
        3. Avoid generic or ambiguous descriptions.
        4. The refined caption should make index 0 clearly stand out from the other indexed objects without using phrases like “distinguished by” or similar expressions.
        4. Do not reference the contour colors or indices directly.
    </outputFormat>"""
    else:
        raise ValueError(f"Invalid stage: {stage}")
    return prompt





