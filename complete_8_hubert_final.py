from os import path
from python_speech_features import delta
from python_speech_features import mfcc
import glob
import numpy as np
import scipy.io.wavfile as wav
import sys
import os
import csv
from tqdm import tqdm
import torch
import torchaudio
import pickle
import complete_utils as cu
import pandas as pd

"""Prediction Constraints in Script:
- Only speech instances containing objects in visual output considered.
- Since instructs limited by above all instances considered for instruction prediction
and then only instructions applicable to predicted object are considered.
- If there is a tie between objects then there frequecies in KNN are multiplied with 
the vision system's predicted confidence.
- Additional feature for the predicted instructions to be multiplied with their 
probabilities to prevent ties."""

# Location of speech_dtw and class_mappings_optimized:
sys.path.append('/Users/ryanmildenhall/Desktop/Masters/Research_Project/dtw_attempt/')
from speech_dtw import qbe
import class_mappings_optimized as cm

# Root working directory:
os.chdir("/Users/ryanmildenhall/Desktop/Masters/Research_Project/complete_implementation_0/yolov5")
sys.path.append("..")
sys.path.append(path.join("..", "utils"))

# Variables and constants:
unseen = True
first_time = True
only_inst_in_im = False
only_inst_for_ob = False
weight_frequencies = False
weight_instruct = False
filt_with_ob = True
num_classes = 50
n_cpus = 4
n_nearest = 5
n_shot = 4
predicted_classes = []
object_not_present = 0
restrict = [only_inst_in_im, only_inst_for_ob, weight_frequencies, weight_instruct, filt_with_ob]

# Set the device depending on resources:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom model checkpoint:
custom_checkpoint_path = './yolov5s6_full_dataset_from_pretrained_all_augs.pt'
vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_checkpoint_path, force_reload=True).to(device)

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(device)
encoder_outputs = []
proj_outputs = []
feat_proj_outputs = []
layer_to_access = 0

# Defining a hook function access specific encoder layer representation:
def hook_func_encoder(module, input, output):
    encoder_outputs.append(output) 

# Doing the same for the projection layer:
def hook_func_proj(module, input, output):
    proj_outputs.append(output) 

# Doing the same for the projection layer:
def hook_func_feat_proj(module, input, output):
    feat_proj_outputs.append(output) 

# Vision Inference
def run_vision_inference(curr_im):
    return vision_model(curr_im)

# Extract Features
def extract_speech_features(wav_fn, cmvn=True):
    """Return the sequential features for an audio file."""
    wav, sr = torchaudio.load(wav_fn)
    assert sr == 16000
    wav = wav.unsqueeze(0).to(device)

    # Register the forward hook in one of the encoder layers:
    hook_handle_encoder = hubert.encoder.layers[layer_to_access].register_forward_hook(hook_func_encoder)
    hook_handle_proj = hubert.proj.register_forward_hook(hook_func_proj)
    hook_handle_feat_proj = hubert.feature_projection.register_forward_hook(hook_func_feat_proj)

    # Extract speech units from final layer of encoder:
    with torch.inference_mode():
        final_out = hubert(wav)
        units = hubert.units(wav)

    # Remove the hooks after the forward pass
    hook_handle_encoder.remove()
    hook_handle_proj.remove()
    hook_handle_feat_proj.remove()

    # Convert features to appropriate data structure to that they can be further processed:
    features = feat_proj_outputs[0].numpy()

    curr_shape = features.shape
    features = features.reshape(curr_shape[1], curr_shape[2])
    if cmvn:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
       
    return features

def get_recordings(query_recordings, train_recordings):
    """Function to load all training and test recordings and then saving them in a list.
    When imported the sequential features for an audio file are extracted."""
    query_mfcc_list = []
    search_mfcc_list = []

    for wav_fn in sorted(glob.glob(path.join(query_recordings, "*.wav"))):
        query_mfcc_list.append(extract_speech_features(wav_fn))

    for wav_fn in sorted(glob.glob(path.join(train_recordings, "*.wav"))):
        search_mfcc_list.append(extract_speech_features(wav_fn))

    return query_mfcc_list, search_mfcc_list

# Reading through folder of audio recordings and labelling audio clips accordingly:
def label_audio_as_tuple(folder_path):
    tuple_labels = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".wav"):
            parts = filename.split('_')
            if len(parts) == 2:
                prefix, number = parts
                number = int(number[0:3])
                label_index = number % num_classes
                tuple_labels.append(class_list[label_index])     
    
    return tuple_labels

if num_classes == 105:
    # Indices of images in test set to remove when testing system on unseen instances:
    indices_to_remove = [11, 21, 25, 27, 34, 35, 43, 53, 56, 62, 67, 76, 99, 104]
    # List of images in test set:
    image_list = ['IMG_0005.jpeg', 'IMG_0006.jpeg', 'IMG_0011.jpeg', 'IMG_0018.jpeg', 'IMG_0033.jpeg', 'IMG_0054.jpeg', 'IMG_0039.jpeg', 'IMG_0040.jpeg', 'IMG_0037.jpeg', 'IMG_0016.jpeg', 'IMG_0017.jpeg', 'IMG_0030.jpeg', 'IMG_0031.jpeg', 'IMG_0015.jpeg', 'IMG_0021.jpeg', 'IMG_0038.jpeg', 'IMG_0046.jpeg', 'IMG_0047.jpeg', 'IMG_0029.jpeg', 'IMG_0032.jpeg', 'IMG_0206.jpeg', 'IMG_0284.jpeg', 'IMG_0354.jpeg', 'IMG_0369.jpeg', 'IMG_0370.jpeg', 'IMG_0385.jpeg', 'IMG_0043.jpeg', 'IMG_0045.jpeg', 'IMG_0358.jpeg', 'IMG_0359.jpeg', 'IMG_0364.jpeg', 'IMG_0034.jpeg', 'IMG_0360.jpeg', 'IMG_0361.jpeg', 'IMG_0378.jpeg', 'IMG_0365.jpeg', 'IMG_0374.jpeg', 'IMG_0377.jpeg', 'IMG_0382.jpeg', 'IMG_0391.jpeg', 'IMG_0392.jpeg', 'IMG_0393.jpeg', 'IMG_5141.jpeg', 'IMG_0022.jpeg', 'IMG_0028.jpeg', 'IMG_0200.jpeg', 'IMG_0279.jpeg', 'IMG_0269.jpeg', 'IMG_0270.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_5051.jpeg', 'IMG_5030.jpeg', 'IMG_5031.jpeg', 'IMG_5123.jpeg', 'IMG_5167.jpeg', 'IMG_0373.jpeg', 'IMG_5052.jpeg', 'IMG_5053.jpeg', 'IMG_5054.jpeg', 'IMG_0185.jpeg', 'IMG_0405.jpeg', 'IMG_0406.jpeg', 'IMG_0407.jpeg', 'IMG_0414.jpeg', 'IMG_0415.jpeg', 'IMG_0416.jpeg', 'IMG_0417.jpeg', 'IMG_0422.jpeg', 'IMG_0423.jpeg', 'IMG_5029.jpeg', 'IMG_5032.jpeg', 'IMG_5095.jpeg', 'IMG_0429.jpeg', 'IMG_0435.jpeg', 'IMG_0436.jpeg', 'IMG_5097.jpeg', 'IMG_0355.jpeg', 'IMG_0356.jpeg', 'IMG_0384.jpeg', 'IMG_0041.jpeg', 'IMG_0140.jpeg', 'IMG_0141.jpeg', 'IMG_0146.jpeg', 'IMG_0048.jpeg', 'IMG_0049.jpeg', 'IMG_0050.jpeg', 'IMG_5038.jpeg', 'IMG_0051.jpeg', 'IMG_0169.jpeg', 'IMG_0170.jpeg', 'IMG_0035.jpeg', 'IMG_0036.jpeg', 'IMG_5017.jpeg', 'IMG_5018.jpeg', 'IMG_0324.jpeg', 'IMG_0325.jpeg', 'IMG_4928.jpeg', 'IMG_4933.jpeg', 'IMG_5021.jpeg', 'IMG_0052.jpeg', 'IMG_0069.jpeg', 'IMG_0070.jpeg', 'IMG_0073.jpeg', 'IMG_0074.jpeg']
    # Load class list and format it appropriately:
    class_file = open("./instruction_files/instructions_tuple.csv", "r")
elif num_classes == 50:
    indices_to_remove = [2, 9, 22, 27, 39, 42, 45]
    image_list = ['IMG_0005.jpeg', 'IMG_0006.jpeg', 'IMG_0354.jpeg', 'IMG_0029.jpeg', 'IMG_0043.jpeg', 'IMG_0015.jpeg', 'IMG_0405.jpeg', 'IMG_0324.jpeg', 'IMG_0391.jpeg', 'IMG_0030.jpeg', 'IMG_0369.jpeg', 'IMG_0037.jpeg', 'IMG_0169.jpeg', 'IMG_0016.jpeg', 'IMG_0048.jpeg', 'IMG_0017.jpeg', 'IMG_0200.jpeg', 'IMG_5029.jpeg', 'IMG_0406.jpeg', 'IMG_0034.jpeg', 'IMG_0018.jpeg', 'IMG_0140.jpeg', 'IMG_0021.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_0035.jpeg', 'IMG_5038.jpeg', 'IMG_0011.jpeg', 'IMG_0038.jpeg', 'IMG_0022.jpeg', 'IMG_0360.jpeg', 'IMG_0382.jpeg', 'IMG_0031.jpeg', 'IMG_0039.jpeg', 'IMG_0364.jpeg', 'IMG_0032.jpeg', 'IMG_0373.jpeg', 'IMG_5030.jpeg', 'IMG_0045.jpeg', 'IMG_0361.jpeg', 'IMG_0355.jpeg', 'IMG_5017.jpeg', 'IMG_0141.jpeg', 'IMG_0028.jpeg', 'IMG_0033.jpeg', 'IMG_0365.jpeg', 'IMG_0040.jpeg', 'IMG_0325.jpeg', 'IMG_5031.jpeg', 'IMG_0051.jpeg']
    class_file = open("./instruction_files/instructions_tuple_50.csv", "r")
else: 
    print("Please enter a valid number of classes.")

class_list = list(csv.reader(class_file, delimiter="\n"))
class_file.close()
all_instruct, all_object = cm.extract_2_classes(class_list)

# Identifying unique object and instruction classes in the dataset:
unique_instructs = list(set(all_instruct))
unique_objects = list(set(all_object))
class_list = [val[0] for val in class_list]
remapped_classes = cm.class_2_to_id(all_instruct, all_object, unique_instructs, unique_objects)
remapping_dict = cu.map_lists_to_dict(unique_objects, cu.vision_objects)

# Fetching the labels for training and test sets:
if num_classes == 105:
    query_recordings_folder = "./instruction_files/query_eng_trunc/"
    query_list = label_audio_as_tuple(query_recordings_folder)
    train_recordings_folder = "./instruction_files/train_eng_trunc/"
    train_list = label_audio_as_tuple(train_recordings_folder)
elif num_classes == 50:
    query_recordings_folder = "./instruction_files/reduced_recordings_ara/test/"
    query_list = label_audio_as_tuple(query_recordings_folder)
    train_recordings_folder = "./instruction_files/reduced_recordings_ara/train/"
    train_list = label_audio_as_tuple(train_recordings_folder)

class_to_id = {class_name: index for index, class_name in enumerate(class_list)}
query_class = np.array([class_to_id[curr_ele] for curr_ele in query_list])
train_class = np.array([class_to_id[curr_ele] for curr_ele in train_list])

# Setting up variables to allow one to filter valid instructions for a particular object:
validity_df = pd.read_csv('./instruction_files/class_to_instruct.csv')
df_instructs = [curr_val.split()[0] for curr_val in list(validity_df.columns)[1:]]
validity_df = validity_df.to_numpy()
df_objects = validity_df[:,0]
validity_df = validity_df[:,1:]

# Calculatiing the probability of the different instructions and remapping them to the correct order:
total_combinations = np.sum(validity_df)
instruct_sums = np.sum(validity_df, axis = 0)
instruct_sums = instruct_sums / total_combinations
instruct_dict = {df_instructs[idx]: instruct_sums[idx] for idx in range(len(df_instructs))}
instruct_probs = [instruct_dict[val] for val in unique_instructs]

#========================================================================================================================#
# Temporary
if first_time == True:
    # Extracting features from signals:
    query_mfcc, train_mfcc = get_recordings(query_recordings_folder, train_recordings_folder)

    query_mfcc = [curr_arr.astype(np.float64) for curr_arr in query_mfcc]
    train_mfcc = [curr_arr.astype(np.float64) for curr_arr in train_mfcc]

    # Saving processed audio files for faster loading in future access:
    with open('query_mfcc_eng_asw.pkl', 'wb') as file:
        pickle.dump(query_mfcc, file)

    with open('train_mfcc_eng_asw.pkl', 'wb') as file:
        pickle.dump(train_mfcc, file)

# Loading pre-processed audio files: 
with open('query_mfcc_eng_asw.pkl', 'rb') as file:
    query_mfcc = pickle.load(file)

with open('train_mfcc_eng_asw.pkl', 'rb') as file:
    train_mfcc = pickle.load(file)
#========================================================================================================================#

full_count = len(train_mfcc)

# Removing previously-specified indices from audio feature lists in the training set:
if unseen == True:
    train_mfcc, train_class = cm.remove_select(indices_to_remove, train_class, train_mfcc, num_classes)

new_count = len(train_mfcc)
unseen_count = full_count - new_count

# Calculate the scores using the parallelised implementation
for idx, curr_query in tqdm(enumerate(query_mfcc)):
    im = "./query_images/" + image_list[idx]
    #========================================================================================================================#
    # Start of vision inference:
    #========================================================================================================================#
    # Access and process the results as needed
    vision_output = run_vision_inference(im)
    # Uncomment next line to view images as they are processed. 
    #vision_output.show()
    output_df = vision_output.pandas().xyxy[0]
    vision_output_ids = [val[1] for val in enumerate(output_df['class'])]
    vision_output_ids_s = [remapping_dict.get(val, 999) for val in vision_output_ids]
    prediction_confidences = [val[1] for val in enumerate(output_df['confidence'])]
    #========================================================================================================================#
    # End of vision inference:
    # Start of speech inference:
    #========================================================================================================================#
    dtw_costs = qbe.parallel_dtw_sweep_min([query_mfcc[idx]], train_mfcc, n_cpus=n_cpus)
    temp_array = np.array(dtw_costs[0])

    if only_inst_in_im == True:
        temp_array_2 = train_class[np.argsort(temp_array)]

        if only_inst_for_ob == True:
            neighbour_indices_ob = cu.lim_speech_to_image(vision_output_ids, temp_array_2, remapping_dict, all_object, unique_objects)
            _, predicted_object_temp = cm.process_nearest_full(neighbour_indices_ob[:n_nearest], all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
            instruct_classes_for_ob = cu.lim_instruct_by_ob(unique_objects[predicted_object_temp[0]], validity_df, df_objects, df_instructs)
            neighbour_indices_in = cu.lim_speech_to_object(instruct_classes_for_ob, neighbour_indices_ob, all_instruct, unique_instructs)[:n_nearest]
            predicted_instruct_temp, _ = cm.process_nearest_full(neighbour_indices_in, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
        else:
            neighbour_indices_ob = cu.lim_speech_to_image(vision_output_ids, temp_array_2, remapping_dict, all_object, unique_objects)[:n_nearest]
            _, predicted_object_temp = cm.process_nearest_full(neighbour_indices_ob, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
            instruct_classes_for_ob = cu.lim_instruct_by_ob(unique_objects[predicted_object_temp[0]], validity_df, df_objects, df_instructs)
            neighbour_indices_in = cu.lim_speech_to_object(instruct_classes_for_ob, temp_array_2, all_instruct, unique_instructs)[:n_nearest]
            predicted_instruct_temp, _ = cm.process_nearest_full(neighbour_indices_in, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
    else:
        neighbour_indices = train_class[np.argsort(temp_array)[:n_nearest]]
        predicted_instruct_temp, predicted_object_temp = cm.process_nearest_full(neighbour_indices, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)

    #========================================================================================================================#
    # End of speech inference.
    # Start of comparison between models
    #========================================================================================================================#
    if filt_with_ob == True:
        found = False
        for idy, val in enumerate(predicted_object_temp):
            id_to_look_for = list({i for i in remapping_dict if remapping_dict[i]==val})[0]
            if id_to_look_for in vision_output_ids:
                found = True
                val_index = vision_output_ids.index(id_to_look_for)
                final_predicted_object = val
                break

        if found == False:
            final_predicted_object = 999
            object_not_present += 1

        predicted_classes.append((predicted_instruct_temp[0], final_predicted_object))
    else:
        predicted_classes.append((predicted_instruct_temp[0], predicted_object_temp[0]))


# Calculating percentage of correct predictions:
if unseen == False:
    final_accs = cm.return_accuracies(remapped_classes, predicted_classes)
else:
    final_accs = cm.return_accuracies_complete(remapped_classes, predicted_classes, indices_to_remove, len(query_mfcc))

print("=================================================================================================")
print("Prediction Accuracy for " + str(n_nearest) + " Nearest Neigbours:")
print("-------------------------------------------------------------------------------------------------")
print("Instructions: " + "Unseen: " + str(final_accs[0][0]) + "%, " + "Seen: " + str(final_accs[0][1]) + "%, " + "Overall: " + str(final_accs[0][2]) + "%")
print("-------------------------------------------------------------------------------------------------")
print("Objects: " + "Unseen: " + str(final_accs[1][0]) + "%, " + "Seen: " + str(final_accs[1][1]) + "%, " + "Overall: " + str(final_accs[1][2]) + "%")
print("-------------------------------------------------------------------------------------------------")
print("Overall: " + "Unseen: " + str(final_accs[2][0]) + "%, " + "Seen: " + str(final_accs[2][1]) + "%, " + "Overall: " + str(final_accs[2][2]) + "%")
print("=================================================================================================")
print("The number of times the predicted speech object class was not present in the image: " + str(object_not_present))
print("=================================================================================================")

# Save results to an output file for further analysis:
with open('output_metrics.csv', 'a') as output_file:
    writer_object = csv.writer(output_file)
    writer_object.writerow(["hubert_encoder_11_1024", "eng_105", n_shot, n_nearest, final_accs[0][0], final_accs[0][1], final_accs[0][2], final_accs[1][0], final_accs[1][1], final_accs[1][2], final_accs[2][0], final_accs[2][1], final_accs[2][2], restrict[0], restrict[1], restrict[2], restrict[3], restrict[4], object_not_present])
    output_file.close()