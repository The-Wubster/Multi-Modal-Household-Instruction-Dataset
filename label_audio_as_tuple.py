import os
import csv

# Root working directory:
#os.chdir("/Volumes/Ryan_Extern/Normal_Dataset_Full/Training_Data/bound_images")
os.chdir("/Users/ryanmildenhall/Desktop/Masters/Research_Project/complete_implementation_0/yolov5/instruction_files")

# List of classes to label audio as based on 50 question interview:
classes = ["Go, Window", "Where, Toilet", "Go, Fridge", "Go, Dustbin", "Open, Oven", "Where, Mirror", "Close, Dishwasher", "Fetch, Computer", "Throw, Cup", "Close, Dustbin", "Open, Fridge", # 10 
 "Go, Couch", "Where, Bed", "Turn, Tap", "Fetch, Pillow", "Where, Cupboard", "Pick, Cloth", "Where, Jug", "Open, Dishwasher", "Close, Microwave", "Go, Sink", "Where, Television", # 21
 "Where, Bottle", "Throw, Pot", "Fetch, Pot", "Open, Draws", "Go, FirePlace", "Open, Window", "Open, Curtains", "Pick, Bottle", "Turn, Microwave", "Go, Toaster", "Go, Table", # 32
 "Where, Light", "Fetch, Kettle", "Open, Cupboard", "Fetch, Bowl", "Pick, Jug", "Turn, Oven", "Where, Microwave", "Where, CoffeeMachine", "Go, Desk", # 41
 "Go, Television", "Where, Sink", "Close, Door", "Pick, Kettle", "Clean, Carpet", "Pick, Computer", "Fetch, Cup", "Throw, Flowers"] #49
num_classes = len(classes)
# Reading through folder of audio recordings and labelling audio clips accordingly:
def label_audio_as_tuple(folder_path):
    tuple_labels = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".wav"):
            parts = filename.split('_')
            if len(parts) == 2:
                prefix, number = parts
                number = int(number[0:3])
                label_index = number % len(classes)
                tuple_labels.append(classes[label_index])     
    
    return tuple_labels
     

print("Start")
output_list = label_audio_as_tuple("./reduced_recordings_ara/test/")
print("End")

# Write the list to a CSV file
with open('instructions_query_ara_tuple.csv', 'w') as csv_file:
    csv_file.write('\n'.join(output_list))