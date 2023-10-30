import numpy as np

speech_objects = ['Fridge', 'Toaster', 'Dishwasher', 'Tap', 'Cloth', 'Cupboard', 'Sink', 'Computer', 'Draws', 'Toilet', #0-9
        'Television', 'Desk', 'Pot', 'Table', 'Cup', 'Bottle', 'Bowl', 'Window', 'Mirror', 'CoffeeMachine', 'Microwave', #10-20 
        'FirePlace', 'Light', 'Pillow', 'Flowers', 'Curtains', 'Door', 'Bed', 'Dustbin', 'Couch', 'Kettle', 'Jug', 'Carpet', #21-32
        'Oven'] # 33
vision_objects = [ 'Window', 'Door', 'Light', 'Carpet', 'Cupboard', 'Mirror', 'Curtains', 'Dustbin', 'Art', 'Fridge', #0-9
         'Oven', 'Microwave', 'Kettle', 'Toaster', 'Tap', 'Sink', 'Pot', 'Cup', 'Bowl', 'Toilet', #10-19
         'Dishwasher', 'Cloth', 'Jug', 'Bottle', 'CoffeeMachine', 'Counter', 'Couch', 'Television', 'Pillow', 'FirePlace', #20-29
         'Chair', 'Table', 'Bed', 'Draws', 'Desk', 'Computer', 'Flowers']

def map_lists_to_dict(original_list, new_list):
    """Takes in two lists and returns an id mapping from new list to original list
    where a key represents the id of an element in the new list and the value 
    represents the id of the element in the original . For example: If one has the
    index of an item in the newlist they can get the id of the value in the original 
    list by getting the value where the key is the current known index. Achieved 
    with: result = dict[key]. To get the reverse: 
    result = list({i for i in dict if dict[i]=="val"})[0]"""

    id_to_id_mapping = {curr_val: i for i, curr_val in enumerate(original_list)}
    return {i: id_to_id_mapping[curr_val] for i, curr_val in enumerate(new_list) if curr_val in original_list}

mapping_dict = map_lists_to_dict(speech_objects, vision_objects)

speech_id = 17

vision_id = list({i for i in mapping_dict if mapping_dict[i]==speech_id})[0]

def lim_speech_to_object(valid_instructs, ordered_neigbours, all_instruct_instances_original, unique_instructs):
    new_neighbours = []  # Initialize an empty list to store new neighbor indices

    instruct_for_ob = list(valid_instructs)  # Create a copy of 'valid_instructs' and convert it to a list
    all_instruct_instances = list(all_instruct_instances_original)  # Create a copy of 'all_instruct_instances_original' and convert it to a list

    # Convert instructions in 'instruct_for_ob' to their indices in 'unique_instructs'
    for idx, curr_val in enumerate(instruct_for_ob):
        instruct_for_ob[idx] = unique_instructs.index(curr_val)

    # Convert all instruction instances in 'all_instruct_instances' to their corresponding indices in 'unique_instructs'
    for idx, curr_val in enumerate(all_instruct_instances):
        all_instruct_instances[idx] = unique_instructs.index(curr_val)

    # Iterate through 'ordered_neigbours' and add the indices to 'new_neighbours' if they exist in 'instruct_for_ob'
    for val in ordered_neigbours:
        if all_instruct_instances[val] in instruct_for_ob:
            new_neighbours.append(val)

    return np.array(new_neighbours)  # Return 'new_neighbours' as a NumPy array

def lim_speech_to_image(vision_output_ids, ordered_neigbours, remapping_dict, all_object_instances_original, unique_objects):
    new_neighbours = []  # Initialize an empty list to store new neighbor indices

    ob_in_im = list(vision_output_ids)  # Create a copy of the 'vision_output_ids' and convert it to a list
    all_object_instances = list(all_object_instances_original)  # Create a copy of 'all_object_instances_original' and convert it to a list

    # Replace values in 'ob_in_im' based on 'remapping_dict' if they exist, otherwise use 999
    for idx, curr_val in enumerate(ob_in_im):
        if curr_val in remapping_dict:
            ob_in_im[idx] = remapping_dict[curr_val]
        else:
            ob_in_im[idx] = 999

    # Convert object instances in 'all_object_instances' to their indices in 'unique_objects'
    for idx, curr_val in enumerate(all_object_instances):
        all_object_instances[idx] = unique_objects.index(curr_val)

    # Iterate through 'ordered_neigbours' and add the indices to 'new_neighbours' if they exist in 'ob_in_im'
    for val in ordered_neigbours:
        if all_object_instances[val] in ob_in_im:
            new_neighbours.append(val)

    return np.array(new_neighbours)  # Return 'new_neighbours' as a NumPy array


def lim_instruct_by_ob(predicted_object, validity_mat, df_objects, df_instructs):
    valid_instructs = []  # Initialize an empty list to store valid instructions for the predicted object

    temp_arr = list(df_objects.copy())  # Create a copy of the 'df_objects' and convert it to a list
    object_id = temp_arr.index(predicted_object)  # Find the index of the 'predicted_object' in the list

    curr_row = validity_mat[object_id, :]  # Extract the row corresponding to 'predicted_object' from 'validity_mat'

    for idx, curr_val in enumerate(curr_row):
        if curr_val == 1:
            valid_instructs.append(df_instructs[idx])  # If the value in the 'validity_mat' is 1, add the corresponding instruction from 'df_instructs' to the 'valid_instructs' list

    return valid_instructs  # Return the list of valid instructions for the predicted object

