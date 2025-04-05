import json

def vg_examine():
    # VG
    source = "/home/cvlab14/project/sangbeom/data3/seg_cap/dataset/visual_genome/annotations/region_descriptions.json"

    with open(source) as f:
        data = json.load(f)

    image_ids = {}
    for item in data:
        for region in item['regions']:
            if region["image_id"] in image_ids:
                image_ids[region["image_id"]].append(region["phrase"])
            else:
                image_ids[region["image_id"]] = [region["phrase"]]

    duplicate = 0
    num_captions = 0
    for image_id in image_ids:
        captions = image_ids[image_id]

        break_bool = False
        for q_id, caption in enumerate(captions):
            num_captions += 1
            for t_id, t_caption in enumerate(captions):
                if caption == t_caption and q_id != t_id:
                    duplicate +=1
            #         break_bool = True
            #         break
            # if break_bool:
            #     break



    # print(f"Number of image = {len(image_ids)}")
    # print(f"Number of duplicated image = {duplicate}")
    # Number of image = 108077
    # Number of duplicated image = 72076

    print(f"Number of Captions = {num_captions}")
    print(f"Number of duplicated Caption = {duplicate}")
    # Number of Captions = 5408689
    # Number of duplicated Caption = 4192138

    print(f"Number of captions per image = {num_captions / len(image_ids)}")
    # Number of captions per image = 50.04477363361307
    breakpoint()
# ------------------------------------ Refcocog ----------------------------

def refcocog_examine():
    # Refcocog
    source = "/home/cvlab14/project/sangbeom/data3/seg_cap/dataset/refcoco/annotations/captions_train2014.json"
    with open(source) as f:
        data = json.load(f)

    image_ids = {}
    for d in data['annotations']:
        if d['image_id'] in image_ids:
            image_ids[d['image_id']].append(d['caption'])
        else:
            image_ids[d['image_id']] = [d['caption']]

    duplicate = 0
    num_captions = 0
    for image_id in image_ids:
        captions = image_ids[image_id]
        break_bool = False
        for q_id, caption in enumerate(captions):
            num_captions += 1

            for t_id, t_caption in enumerate(captions):
                if caption == t_caption and q_id != t_id:
                    duplicate += 1
                    break
            #         print(captions, caption)
            #         break_bool = True
            #         break
            # if break_bool:
            #     break

    # print(f"Number of image = {len(image_ids)}")
    # print(f"Number of duplicated image = {duplicate}")
    # # Number of image = 82783
    # # Number of duplicated image = 102

    print(f"Number of Captions = {num_captions}")
    print(f"Number of duplicated Caption = {duplicate}")
    # Number of Captions = 414113
    # Number of duplicated Caption = 205

    print(f"Number of captions per image = {num_captions/len(image_ids)}")
    # Number of captions per image = 5.0023917954169335
    breakpoint()

if __name__ == '__main__':
    # refcocog_examine()
    vg_examine()