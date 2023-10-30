from sys import stdlib_module_names
import numpy as np
import matplotlib.pyplot as plt

# Function to extract instructs and objects from a list of tuples
def extract_2_classes(curr_list):
    objects_ = []
    instructs= []
    for val in curr_list:
        instruct, obj = val[0].split(", ")
        instructs.append(instruct)
        objects_.append(obj)

    return instructs, objects_

# Function to map instructs and objects to IDs
def class_2_to_id(curr_instruct, curr_object, unique_instructs, unique_objects):
    instruct_index_map = {val: idx for idx, val in enumerate(unique_instructs)}
    object_index_map = {val: idx for idx, val in enumerate(unique_objects)}

    instruct_id = [instruct_index_map[val] for val in curr_instruct]
    object_id = [object_index_map[val] for val in curr_object]

    curr_mapped = list(zip(instruct_id, object_id))

    return curr_mapped

# Function to map IDs back to instructs and objects
def id_to_2_class(curr_2_id, unique_instructs, unique_objects):
    instructs = [unique_instructs[val[0]] for val in curr_2_id]
    objects = [unique_objects[val[1]] for val in curr_2_id]

    return list(zip(instructs, objects))

# Function to extract the most frequent element from a list
def extract_frequent(curr_list):
    unique, counts = np.unique(curr_list, return_counts=True)
    return unique[np.argmax(counts)]

# Function to extract order of the most frequent elements from a list:
def extract_frequent_full(curr_list):
    unique, counts = np.unique(curr_list, return_counts=True)
    sort_ids = np.argsort(-counts)
    unique = unique[sort_ids]
    counts = counts[sort_ids]
    return unique, counts

# Function to process the nearest neighbors and return the most frequent instruct and object
def process_nearest(predicted_list, all_instructs, all_objects):
    predicted_in = []
    predicted_ob = []

    for val in predicted_list:
        predicted_in.append(all_instructs[val])
        predicted_ob.append(all_objects[val])

    final_in = extract_frequent(predicted_in)
    final_ob = extract_frequent(predicted_ob) 

    return (final_in, final_ob)

# Function to process the nearest neighbors and returns lists of the most frequent instructs and objects in order.
def process_nearest_full(predicted_list, all_instructs, all_objects, unique_instructs, unique_objects, vision_confidences, add_confidence, vision_output_ids_s, weight_instruct, instruct_probs):
    predicted_in = []
    predicted_ob = []

    for val in predicted_list:
        temp_res = class_2_to_id([all_instructs[val]], [all_objects[val]], unique_instructs, unique_objects)
        predicted_in.append(temp_res[0][0])
        predicted_ob.append(temp_res[0][1])

    if add_confidence == True:
        final_in, in_counts = extract_frequent_full(predicted_in)
        final_ob, ob_counts = extract_frequent_full(predicted_ob)
        in_counts = np.array(in_counts, dtype='float64')
        ob_counts = np.array(ob_counts, dtype='float64')

        for idx, val in enumerate(final_ob):
            if val in vision_output_ids_s:
                ob_counts[idx] = np.float64(ob_counts[idx]) * np.float64(vision_confidences[vision_output_ids_s.index(val)])
            else:
                ob_counts[idx] = 0
        
        sort_ids = np.argsort(-ob_counts)
        final_ob = final_ob[sort_ids]
        ob_counts = ob_counts[sort_ids]

        if weight_instruct == True:
            for idx, val in enumerate(final_in):
                in_counts[idx] = np.float64(in_counts[idx]) * np.float64(instruct_probs[val])

            sort_ids = np.argsort(-in_counts)
            final_in = final_in[sort_ids]
            in_counts = in_counts[sort_ids]

    else:
        final_in, _ = extract_frequent_full(predicted_in)
        final_ob, _ = extract_frequent_full(predicted_ob) 

    return final_in, final_ob

# Function to process the nearest neighbors and returns lists of the most frequent instructs and objects in order.
def process_nearest_full_single(predicted_list, all_instructs, all_objects, unique_instructs, unique_objects):
    predicted_in = []
    #predicted_ob = []

    for val in predicted_list:
        temp_res = class_2_to_id([all_instructs[val]], [all_objects[val]], unique_instructs, unique_objects)
        predicted_in.append(temp_res[0][0])
        #predicted_ob.append(temp_res[0][1])

    
    final_in, _ = extract_frequent_full(predicted_in)
    #final_ob = extract_frequent_full(predicted_ob) 

    return final_in

# Function to calculate accuracy
def return_accuracies(actual_class, predicted_class):
    object_correct = 0
    instruct_correct = 0
    overall_correct = 0
    for idx, val in enumerate(predicted_class):
        flag = 0
        if val[0] == actual_class[idx][0]:
            instruct_correct += 1
            flag += 1
        if val[1] == actual_class[idx][1]:
            object_correct += 1
            flag += 1
        if flag == 2:
            overall_correct += 1

    total = len(predicted_class)
    in_acc = round(((instruct_correct / total) * 100), 2)
    ob_acc = round(((object_correct / total) * 100), 2)
    overall_acc = round(((overall_correct / total) * 100), 2)
    
    return in_acc, ob_acc, overall_acc

# Function to calculate accuraccies for seen, unseen and combined sections of data:
def return_accuracies_complete(actual_class, predicted_class, unseen_indices, num_query):
    object_correct_unseen = 0
    instruct_correct_unseen = 0
    overall_correct_unseen = 0
    object_correct_seen = 0
    instruct_correct_seen = 0
    overall_correct_seen = 0
    total = len(predicted_class)
    # Required to change line of code below if test set changes from standard 105 classes:
    num_unseen = len(unseen_indices) * int(round(total / num_query))
    num_seen = total - num_unseen

    for idx, val in enumerate(predicted_class):
        flag = 0
        if idx in unseen_indices:
            if val[0] == actual_class[idx][0]:
                instruct_correct_unseen += 1
                flag += 1
            else: 
                print(str(idx) + "    Incorrect instruct: Predicted instruct: " + str(val[0]) + " Actual instruct: " + str(actual_class[idx][0]))
            if val[1] == actual_class[idx][1]:
                object_correct_unseen += 1
                flag += 1
            else: 
                print(str(idx) + "    Incorrect object: Predicted object: " + str(val[1]) + " Actual object: " + str(actual_class[idx][1]))
            if flag == 2:
                overall_correct_unseen += 1
        else:
            if val[0] == actual_class[idx][0]:
                instruct_correct_seen += 1
                flag += 1
            if val[1] == actual_class[idx][1]:
                object_correct_seen += 1
                flag += 1
            if flag == 2:
                overall_correct_seen += 1

    in_acc = np.array([round(((instruct_correct_unseen / num_unseen) * 100), 2), round(((instruct_correct_seen / num_seen) * 100), 2), round((((instruct_correct_unseen + instruct_correct_seen) / total) * 100), 2)])
    ob_acc = np.array([round(((object_correct_unseen / num_unseen) * 100), 2), round(((object_correct_seen / num_seen) * 100), 2), round((((object_correct_unseen + object_correct_seen) / total) * 100), 2)])
    overall_acc = np.array([round(((overall_correct_unseen / num_unseen) * 100), 2), round(((overall_correct_seen / num_seen) * 100), 2), round((((overall_correct_unseen + overall_correct_seen) / total) * 100), 2)])
   
    return in_acc, ob_acc, overall_acc

def remove_select(list_indices, indice_list, list_to_alter):
    indices_to_remove = [idx for idx, val in enumerate(indice_list) if val in list_indices]
    altered_indice_list = [indice_list[idx] for idx in range(len(indice_list)) if idx not in indices_to_remove]
    altered_list = [list_to_alter[idx] for idx in range(len(list_to_alter)) if idx not in indices_to_remove]

    return altered_list, np.array(altered_indice_list)

def plot_counts(curr_list):
    my_array = np.array(curr_list)

    unique_elements, counts = np.unique(my_array, return_counts=True)

    sorted_indices = np.argsort(-counts)
    unique_elements = unique_elements[sorted_indices]
    counts = counts[sorted_indices]

    plt.bar(unique_elements, counts)
    plt.xlabel('Unique Elements')
    plt.ylabel('Counts')
    plt.title('Counts of Unique Elements in List')
    plt.xticks(rotation=85)

    plt.show()

    return