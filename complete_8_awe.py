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
import tsne_representation as tr
import pandas as pd
from matplotlib import pyplot as plt

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
filt_with_ob = False
plot_tsne = False
awe_type = "MFCC" #"Hub"
# Indices of images in test set to remove when testing system on unseen instances:
indices_to_remove = [11, 21, 25, 27, 34, 35, 43, 53, 56, 62, 67, 76, 99, 104]
# List of images in test set:
image_list = ['IMG_0005.jpeg', 'IMG_0006.jpeg', 'IMG_0011.jpeg', 'IMG_0018.jpeg', 'IMG_0033.jpeg', 'IMG_0054.jpeg', 'IMG_0039.jpeg', 'IMG_0040.jpeg', 'IMG_0037.jpeg', 'IMG_0016.jpeg', 'IMG_0017.jpeg', 'IMG_0030.jpeg', 'IMG_0031.jpeg', 'IMG_0015.jpeg', 'IMG_0021.jpeg', 'IMG_0038.jpeg', 'IMG_0046.jpeg', 'IMG_0047.jpeg', 'IMG_0029.jpeg', 'IMG_0032.jpeg', 'IMG_0206.jpeg', 'IMG_0284.jpeg', 'IMG_0354.jpeg', 'IMG_0369.jpeg', 'IMG_0370.jpeg', 'IMG_0385.jpeg', 'IMG_0043.jpeg', 'IMG_0045.jpeg', 'IMG_0358.jpeg', 'IMG_0359.jpeg', 'IMG_0364.jpeg', 'IMG_0034.jpeg', 'IMG_0360.jpeg', 'IMG_0361.jpeg', 'IMG_0378.jpeg', 'IMG_0365.jpeg', 'IMG_0374.jpeg', 'IMG_0377.jpeg', 'IMG_0382.jpeg', 'IMG_0391.jpeg', 'IMG_0392.jpeg', 'IMG_0393.jpeg', 'IMG_5141.jpeg', 'IMG_0022.jpeg', 'IMG_0028.jpeg', 'IMG_0200.jpeg', 'IMG_0279.jpeg', 'IMG_0269.jpeg', 'IMG_0270.jpeg', 'IMG_5049.jpeg', 'IMG_5050.jpeg', 'IMG_5051.jpeg', 'IMG_5030.jpeg', 'IMG_5031.jpeg', 'IMG_5123.jpeg', 'IMG_5167.jpeg', 'IMG_0373.jpeg', 'IMG_5052.jpeg', 'IMG_5053.jpeg', 'IMG_5054.jpeg', 'IMG_0185.jpeg', 'IMG_0405.jpeg', 'IMG_0406.jpeg', 'IMG_0407.jpeg', 'IMG_0414.jpeg', 'IMG_0415.jpeg', 'IMG_0416.jpeg', 'IMG_0417.jpeg', 'IMG_0422.jpeg', 'IMG_0423.jpeg', 'IMG_5029.jpeg', 'IMG_5032.jpeg', 'IMG_5095.jpeg', 'IMG_0429.jpeg', 'IMG_0435.jpeg', 'IMG_0436.jpeg', 'IMG_5097.jpeg', 'IMG_0355.jpeg', 'IMG_0356.jpeg', 'IMG_0384.jpeg', 'IMG_0041.jpeg', 'IMG_0140.jpeg', 'IMG_0141.jpeg', 'IMG_0146.jpeg', 'IMG_0048.jpeg', 'IMG_0049.jpeg', 'IMG_0050.jpeg', 'IMG_5038.jpeg', 'IMG_0051.jpeg', 'IMG_0169.jpeg', 'IMG_0170.jpeg', 'IMG_0035.jpeg', 'IMG_0036.jpeg', 'IMG_5017.jpeg', 'IMG_5018.jpeg', 'IMG_0324.jpeg', 'IMG_0325.jpeg', 'IMG_4928.jpeg', 'IMG_4933.jpeg', 'IMG_5021.jpeg', 'IMG_0052.jpeg', 'IMG_0069.jpeg', 'IMG_0070.jpeg', 'IMG_0073.jpeg', 'IMG_0074.jpeg']
n_cpus = 4
n_nearest = 10
n_shot = 10
predicted_classes = []
object_not_present = 0
restrict = [only_inst_in_im, only_inst_for_ob, weight_frequencies, weight_instruct, filt_with_ob]

# Set the device depending on resources:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(device)

# Extract Features
def extract_speech_features(wav_fn, cmvn=True):
    """Return the acoustic word embeddings for an audio file."""
    #if awe_type == "Hub":
    # Using Embeddings from HuBERT Model
    """wav, sr = torchaudio.load(wav_fn)
    assert sr == 16000
    wav = wav.unsqueeze(0).to(device)

    # Extract speech units
    with torch.inference_mode():
        features = hubert.units(wav)"""
    #elif awe_type == "MFCC":
    # Using MFCCs and other speech features:
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    (rate, signal) = wav.read(wav_fn)
    mfcc_static = mfcc(signal, rate)
    mfcc_deltas = delta(mfcc_static, 2)
    mfcc_delta_deltas = delta(mfcc_deltas, 2)
    features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
    if cmvn:
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        features = torch.Tensor(features).unsqueeze(0)

    mean_pool = torch.nn.AvgPool2d((features.size()[1], 1), stride=(features.size()[1], 1))
    features = mean_pool(features)

    features = features.numpy()
    features = features.reshape(1, features.shape[2])
    return features

def get_recordings():
    """Function to load all training and test recordings and then saving them in a list.
    When imported the sequential features for an audio file are extracted."""
    query_mfcc_list = []
    search_mfcc_list = []

    for wav_fn in sorted(glob.glob(path.join("./instruction_files/query_eng_trunc", "*.wav"))):
        query_mfcc_list.append(extract_speech_features(wav_fn))

    for wav_fn in sorted(glob.glob(path.join("./instruction_files/train_eng_trunc", "*.wav"))):
        search_mfcc_list.append(extract_speech_features(wav_fn))

    return query_mfcc_list, search_mfcc_list


# Load class list and format it appropriately:
class_file = open("./instruction_files/instructions_tuple.csv", "r")
class_list = list(csv.reader(class_file, delimiter="\n"))
class_file.close()
all_instruct, all_object = cm.extract_2_classes(class_list)

# Identifying unique object and instruction classes in the dataset:
unique_instructs = list(set(all_instruct))
unique_objects = list(set(all_object))
class_list = [val[0] for val in class_list]
remapped_classes = cm.class_2_to_id(all_instruct, all_object, unique_instructs, unique_objects)
remapping_dict = cu.map_lists_to_dict(unique_objects, cu.vision_objects)

# Fetching the labels for training and test sets for during development:
class_file = open("./instruction_files/instructions_query_eng_tuple.csv", "r")
query_list = list(csv.reader(class_file, delimiter="\n"))
class_file.close()
query_list = [val[0] for val in query_list]

class_file = open("./instruction_files/instructions_train_eng_tuple.csv", "r")
train_list = list(csv.reader(class_file, delimiter="\n"))
class_file.close()
train_list = [val[0] for val in train_list]

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
    query_mfcc, train_mfcc = get_recordings()

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

if plot_tsne == True:
    temp_arr = np.stack(query_mfcc.copy())
    temp_shape = temp_arr.shape
    temp_arr = np.reshape(temp_arr, (temp_shape[0], temp_shape[2]))
    tr.create_2d_tsne(temp_arr, all_instruct)

full_count = len(train_mfcc)

# Removing previously-specified indices from audio feature lists in the training set:
if unseen == True:
    train_mfcc, train_class = cm.remove_select(indices_to_remove, train_class, train_mfcc)

new_count = len(train_mfcc)
unseen_count = full_count - new_count

#========================================================================================================================#
# Load your custom model checkpoint:
custom_checkpoint_path = './yolov5s6_full_dataset_from_pretrained_all_augs.pt' 
vision_model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_checkpoint_path, force_reload=True).to(device)

# Vision Inference
def run_vision_inference(curr_im):
    return vision_model(curr_im)
#========================================================================================================================#    

# Calculate the scores using the parallelised implementation
for idx, curr_query in tqdm(enumerate(query_mfcc)):
    im = "./query_images/" + image_list[idx]
    #========================================================================================================================#
    # Start of vision inference:
    #========================================================================================================================#
    # Access and process the results as needed
    vision_output = run_vision_inference(im)
    output_df = vision_output.pandas().xyxy[0]
    vision_output_ids = [val[1] for val in enumerate(output_df['class'])]
    vision_output_ids_s = [remapping_dict.get(val, 999) for val in vision_output_ids]
    prediction_confidences = [val[1] for val in enumerate(output_df['confidence'])]
    #========================================================================================================================#
    # End of vision inference:
    # Start of speech inference:
    #========================================================================================================================#
    dtw_costs = qbe.parallel_dtw_sweep_min_awe([query_mfcc[idx]], train_mfcc, n_cpus=n_cpus)
    temp_array = np.array(dtw_costs)
    temp_array = temp_array.reshape(temp_array.shape[1])

    if only_inst_in_im == True:
        temp_array_2 = train_class[np.argsort(temp_array)]
        if only_inst_for_ob == True:
            neighbour_indices_ob = cu.lim_speech_to_image(vision_output_ids, temp_array_2, remapping_dict, all_object, unique_objects)
            _, predicted_object_temp = cm.process_nearest_full(neighbour_indices_ob[::-1][:n_nearest], all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
            instruct_classes_for_ob = cu.lim_instruct_by_ob(unique_objects[predicted_object_temp[0]], validity_df, df_objects, df_instructs)
            neighbour_indices_in = cu.lim_speech_to_object(instruct_classes_for_ob, neighbour_indices_ob, all_instruct, unique_instructs)[::-1][:n_nearest]
            predicted_instruct_temp, _ = cm.process_nearest_full(neighbour_indices_in, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
        else:
            neighbour_indices_ob = cu.lim_speech_to_image(vision_output_ids, temp_array_2, remapping_dict, all_object, unique_objects)[::-1][:n_nearest]
            _, predicted_object_temp = cm.process_nearest_full(neighbour_indices_ob, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
            instruct_classes_for_ob = cu.lim_instruct_by_ob(unique_objects[predicted_object_temp[0]], validity_df, df_objects, df_instructs)
            neighbour_indices_in = cu.lim_speech_to_object(instruct_classes_for_ob, temp_array_2, all_instruct, unique_instructs)[::-1][:n_nearest]
            predicted_instruct_temp, _ = cm.process_nearest_full(neighbour_indices_in, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
    else:
        neighbour_indices = train_class[np.argsort(temp_array)[::-1][:n_nearest]]
        predicted_instruct_temp, predicted_object_temp = cm.process_nearest_full(neighbour_indices, all_instruct, all_object, unique_instructs, unique_objects, prediction_confidences, weight_frequencies, vision_output_ids_s, weight_instruct, instruct_probs)
        
    if idx in indices_to_remove:
        print(str(idx) + ' instruct: ' + str(predicted_instruct_temp) + " object: "+ str(predicted_object_temp))
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
    print("=================================================================================================")
    print("Prediction Accuracy for " + str(n_nearest) + " Nearest Neigbours:")
    print("-------------------------------------------------------------------------------------------------")
    print("Instructions: " + str(final_accs[0]) + "% ")
    print("-------------------------------------------------------------------------------------------------")
    print("Objects: " + str(final_accs[1]) + "% ")
    print("-------------------------------------------------------------------------------------------------")
    print("Overall: " + str(final_accs[2]) + "% ")
    print("=================================================================================================")
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

with open('output_metrics.csv', 'a') as output_file:
    if unseen == True:
        writer_object = csv.writer(output_file)
        writer_object.writerow(["hubert_awe_256", "eng_105", n_shot, n_nearest, final_accs[0][0], final_accs[0][1], final_accs[0][2], final_accs[1][0], final_accs[1][1], final_accs[1][2], final_accs[2][0], final_accs[2][1], final_accs[2][2], restrict[0], restrict[1], restrict[2], restrict[3], restrict[4], object_not_present])
        output_file.close()