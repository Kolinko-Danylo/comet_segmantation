import cv2
import os
import numpy as np
import json


def binarize_masks(path):
    for f in os.listdir(path):
      current_img = cv2.imread(path + f)
      current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
      current_img[np.where((current_img==[0, 0, 128]).all(axis=2))] = np.array([0, 0, 0])

      current_img[np.where(((current_img!=[0, 0, 0]) & (current_img!=[255, 255, 255])).any(axis=2))] = np.array([128, 0, 0])
      current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(path+f, current_img)


def convert_to_new_ext(path, ext=".png"):
    for f in os.listdir(path):
        if f[-4:] != ext:
            # print(path+f)
            # print((path+f)[:-4] + ext)
            c = cv2.imread(path + f)
            cv2.imwrite((path + f)[:-4] + ext, c)
            os.remove(path + f)

def draw_borders(json_path, images_path):
    with open(json_path, "r+") as f:
        data = json.loads(f.read())
        cat = data['categories']

        for instance in data['annotations']:
            instance_img = list(filter(lambda x: x['id'] == instance['image_id'], data['images'] ))[0]['file_name']
            instance_cat = list(filter(lambda x: x['id'] == instance['category_id'], cat ))[0]
            mask_filename = instance_img[:-4] + ".png"
            # print(mask_filename)
            current_img = cv2.imread(images_path + mask_filename)
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
            seg = instance['segmentation'][0]
            border = np.array([seg[i:i+2] for i in range(0, len(seg), 2)], dtype=np.int32).reshape(-1, 1, 2)
            # print(border.shape)
            if instance_cat['name'] != 'debris':
                res_img = draw_polygon(current_img, border, thickness=2)
                cv2.imwrite(images_path + mask_filename, res_img)




def draw_polygon(img, coords, thickness):
    color = (255, 255, 255)
    current_img = img.copy().astype(np.uint8)
    current_img = cv2.polylines(current_img, [coords], True, color, thickness=thickness)

    current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
    return current_img

draw_borders("data/test/hicomet_data.json", "data/test/masks/")
binarize_masks("data/test/masks/")
# convert_to_new_ext("data/test/images/")

