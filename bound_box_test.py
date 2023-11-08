# Import required libraries
import cv2
import os 
import csv
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm 

# Set the working directory
os.chdir("/Volumes/Ryan_Extern/Normal_Dataset_Full/Training_Data/")

# Define a function to construct a list of classes from a file
def construct_class_list():
    class_file = open("labels/labels.txt", "r")
    class_data = class_file.read()
    class_list = class_data.split("\n")
    class_list = [curr_element.replace(' ', '') for curr_element in class_list]
    class_list = class_list[:-1]
    class_file.close()
    return class_list

# Define a function to extract an ID from an instruction
def extract_id(instruct_, class_list):
    temp_id = str(instruct_)[2:-3].split()[-1]
    temp_id = class_list.index(temp_id)
    return temp_id

# Read CSV file with instructions
file = open("instruction_100_mod.csv", "r")
instruct_list = list(csv.reader(file, delimiter=",")) 
file.close()

# Define Variables
class_ids = []
image_names = []
final_instructs = []
found_ = 0
 # Create a list of classes from a file
class_list = construct_class_list()
label_list = os.listdir("labels")
label_list = label_list[2:]

# Iterate through instructions and process images
for idx, val in tqdm(enumerate(instruct_list)):
    final_instructs.append(str("Provide the following instruction: ") + ' '.join(str(val)[2:-3].split()))  # Create instruction text
    class_ids.append(extract_id(val, class_list))  # Extract class ID
    found_ = 0
    image_count = 0
    while found_ == 0:
        label_path = "labels/" + label_list[image_count]  # Get label file path
        image_path = "images/" + label_list[image_count][0:8] + ".jpeg"  # Get image file path
        label_df = pd.read_csv(label_path, sep=" ", names=['Item_id', 'x_center', 'y_center', 'x_width', 'y_width'])  # Read label data
        if (class_ids[idx] in label_df['Item_id'].values) and (len(label_df) > 5):
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read image
            height, width, color_ = img.shape
            curr_row = label_df.loc[label_df['Item_id'] == class_ids[idx]].values.flatten().tolist()
            if (curr_row[3] > 0.1) and (curr_row[4] > 0.1):
                x_min = np.maximum(0, round((curr_row[1] - (curr_row[3]/2)) * width))
                x_max = np.minimum(width, round((curr_row[1] + (curr_row[3]/2)) * width))
                y_min = np.maximum(0, round((curr_row[2] - (curr_row[4]/2)) * height))
                y_max = np.minimum(height, round((curr_row[2] + (curr_row[4]/2)) * height))
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 20)  # Draw a rectangle on the image
                cv2.imwrite("test_images_50/" + label_list[image_count][0:8] + ".jpeg", img)  # Save the modified image
                image_names.append(label_list[image_count][0:8] + ".jpeg")
                found_ = 1
                label_list = label_list[:image_count] + label_list[(image_count + 1):]  # Remove the processed label from the list
        image_count += 1

# Ouput processed files
print(image_names)
print(final_instructs)
