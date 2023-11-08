# Multi-Modal-Household-Instruction-Dataset
In this repository, we provide code for the data-capturing application we developed in tkinter, the model scripts, utility files and links to the datasets.

## Introduction
This repository contains the code and links to the datasets used in a "Multi-modal Few-shot Learning in a Household Robot". In the project a multi-modal household robotic system was developed which allows a user to interact with common household objects in a language of their choice by giving a robot a simple instruction to complete. The system was designed to perform well on lower-resource languages. As part of this project, we also collected a dataset of paired images and speech recordings where the verbal instruction is invoked by the user being shown a visual stimulus. The links to the speech and vision datasets are added later on. 

## Description
### bound_box_test
This script processes a list of instructions and images by extracting class IDs from the instructions and searching for corresponding images based on these IDs. For each match, a bounding box is added to an image before the image is saved. This code also constructs modified instruction text.

### Files beginning with "complete"
This code is designed to generate audio-to-text predictions by combining speech and vision data. The model used for audio embeddings can be changed, and the code also includes a comparison between different models. These scripts use a combination of audio features and object predictions from a vision system to make audio predictions based on specified constraints.

The process used in these scripts can be summarised as follows: 
1. Import necessary libraries and modules, including audio processing, vision, and machine learning components. Set rule-based system and model configuration parameters and variables.

2. Load the audio and class data and prepare it for further processing, including removing specific instances if needed. The code distinguishes between "unseen" and "seen" cases.

3. Define functions for speech feature extraction, vision inference, and parallel DTW-based speech-to-speech comparisons. The code also includes rules for filtering valid instructions and objects.

4. Iterate through audio recordings and perform vision inference for each image, followed by speech inference to predict audio instructions. The code also filters predictions based on specified constraints.

5. Calculate and print prediction accuracy, and save the results in a CSV file. The code can handle both "unseen" and "seen" scenarios and records the number of times the predicted speech object class was not present in the image.

The models used in these scripts are based on code in the following repositories:
- [DTW for Speech Processing](https://github.com/kamperh/speech_dtw.git "DTW for Speech Processing")
- [Voice Conversion With Just Nearest Neighbors](http://www.google.fr/](https://github.com/bshall/knn-vc.git) "Voice Conversion With Just Nearest Neighbors")
- [Soft Speech Units for Improved Voice Conversion](http://www.google.fr/](https://github.com/bshall/soft-vc.git) "Soft Speech Units for Improved Voice Conversion")

### generate_questions
This script generates a set of questions based on data from a CSV file. It imports necessary libraries, reads the data, and then iterates through the data to create questions by combining instructions and class names. The programme ensures that all questions are unique and then saves the generated questions to a new CSV file called "instructions.csv."

### lable_audio_as_tuple
The code begins by defining a list of classes for labelling audio clips based on interview questions. It contains a function called `label_audio_as_tuple`, which reads audio files in a specified folder, extracts numerical parts from the filenames, and assigns corresponding class labels based on a modulo operation with the number of classes. The code then executes this function on a specific folder and writes the resulting list of labels to a CSV file named 'instructions_query_ara_tuple.csv'. This code is primarily used for labelling audio clips based on filenames and saving the labelled data in a CSV format.

### complete_utils
This Python code defines a set of functions for the "complete..." scripts. The code first creates two lists, `speech_objects` and `vision_objects`, which represent different objects related to speech and vision, respectively. It then defines a function `map_lists_to_dict` that generates a mapping from one list to another based on the order of elements in the two lists. The `map_lists_to_dict` function takes two lists as input, and it returns a dictionary where keys are the indices of elements in the second list, and values are the corresponding indices of elements in the first list based on their order. The code also includes functions for mapping speech to objects and speech to images. These functions take input data, including instructions and objects, and use the mapping generated earlier to determine valid instructions and objects.

### interview_gui_1
This script defines a Tkinter interface for creating a user interface. The application serves as an interactive tool for collecting descriptions of images and audio recordings based on the presented images. The descriptions are saved to a text file for further analysis or use.

The code can be summarised with the following steps:
1. It imports necessary libraries such as `tkinter`, `PIL` for image processing, `sounddevice` for audio recording, and `numpy` for numerical operations.

2. The code sets up a Tkinter application named `ImageDescriptionApp`, which displays images, collects textual and audio descriptions, and saves the descriptions to a text file.

3. The application allows users to provide instructions for completing common household tasks based on displayed images. The user interface displays images and a text box for entering descriptions. Users can also record audio descriptions.

4. The code defines a list of image filenames and corresponding instructions. Users can navigate through these images and provide instructions for each.

   
### class_mappings_optimized
This Python code defines a set of functions for the "complete..." scripts and allows for the extraction and mapping of instructions and objects to unique IDs and the reverse process as well. Additionally, it provides functions to calculate accuracies and breakdowns for seen and unseen data sections. The code also includes a function for generating bar plots to visualize the distribution of unique elements in a list. These functions are designed for data processing and analysis.

### vad_audio_process
This code is designed to remove silence from audio files stored in a directory. It uses the Silero VAD (Voice Activity Detection) model to extract speech segments from audio files. The code reads audio files, applies the VAD model to detect speech, and then saves the extracted speech segments into a new directory. The performs this process for each file in the provided folder. The original work can be found at [Silero VAD](https://github.com/snakers4/silero-vad.git "Silero VAD").

## Dataset and Model Weights
The vision dataset and weights of the top 5 YOLO models implemented in this project can be found at: https://drive.google.com/drive/folders/10hJtw0Jq2xcDIj-gWoOCALfQXy7sF56H?usp=share_link

The speech dataset is currently in the process of being published. Once this is complete a link to access it will be placed here.

## Environment Setup
To setup the environment for the object detector refer to: https://github.com/The-Wubster/Skripsie.git

## License

Copyright (c) 2022 Stellenbosch University

This data is released under a Creative Commons Attribution-ShareAlike 
license
(<http://creativecommons.org/licenses/by-sa/4.0/>).
