import cv2
import os 
import csv
from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
from tqdm import tqdm

os.chdir("/Volumes/Ryan_Extern/Normal_Dataset_Full/Training_Data/")

def construct_class_list():
    class_file = open("labels/labels.txt", "r")
    class_data = class_file.read()
    class_list = class_data.split("\n")
    class_list = [curr_element.replace(' ', '') for curr_element in class_list]
    class_list = class_list[:-1]
    class_file.close()
    return class_list

def extract_id(instruct_, class_list):
    temp_id = str(instruct_)[2:-3].split()[-1]
    temp_id = class_list.index(temp_id)
    return temp_id

file = open("test.csv", "r")
instruct_list = list(csv.reader(file, delimiter=","))
file.close()

class_ids = []
image_names = []
final_instructs = []
found_ = 0
class_list = construct_class_list()
label_list = os.listdir("labels")
label_list = label_list[2:]

for idx, val in tqdm(enumerate(instruct_list)):
    final_instructs.append(' '.join(str(val)[2:-3].split()[:-1]) + str(" highlighted object"))
    class_ids.append(extract_id(val, class_list))
    found_ = 0
    image_count = 0
    while found_ == 0:
        label_path = "labels/" + label_list[image_count]
        image_path = "images/" + label_list[image_count][0:8] + ".jpeg"
        label_df = pd.read_csv(label_path, sep=" ", names=['Item_id', 'x_center', 'y_center', 'x_width', 'y_width'])
        if (class_ids[idx] in label_df['Item_id'].values):
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            height, width, color_ = img.shape
            curr_row = label_df.loc[label_df['Item_id'] == class_ids[idx]].values.flatten().tolist()
            if (curr_row[3] > 0.1) and (curr_row[4] > 0.1):
                x_min = np.maximum(0, round((curr_row[1] - (curr_row[3]/2)) * width))
                x_max = np.minimum(width, round((curr_row[1] + (curr_row[3]/2)) * width))
                y_min = np.maximum(0, round((curr_row[2] - (curr_row[4]/2)) * height))
                y_max = np.minimum(height, round((curr_row[2] + (curr_row[4]/2)) * height))
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 20)
                cv2.imwrite("bound_images/" + label_list[image_count][0:8] + ".jpeg", img)
                image_names.append(label_list[image_count][0:8] + ".jpeg")
                found_ = 1
                label_list = label_list[:image_count] + label_list[(image_count + 1):]
        image_count += 1

print(image_names)
print(final_instructs)
