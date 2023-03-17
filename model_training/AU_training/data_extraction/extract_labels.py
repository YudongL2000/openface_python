import os
import json

def extract_label_file(label_file_path):
    file_reader = open(label_file_path, "r")
    file_lines = file_reader.readlines()
    label_dict = {}
    for line in file_lines:
        row_id = int(line.split(",")[0])
        au_intensity = int(line.split(",")[1])
        label_dict[row_id] = au_intensity
    file_reader.close()
    return label_dict

def extract_label_list(label_file_path):
    file_reader = open(label_file_path, "r")
    file_lines = file_reader.readlines()
    result_list = []
    for line in file_lines:
        row_id = int(line.split(",")[0])
        au_intensity = int(line.split(",")[1])
        result_list.append(au_intensity)
    file_reader.close()
    return result_list

def extract_filename_no_ext(file_path):
    head, tail = os.path.split(file_path)
    filename_no_ext = os.path.splitext(tail)[0]
    return filename_no_ext


def extract_labels(label_path, preprocess_label_path, vid_name):
    action_unit_path = os.path.join(label_path, vid_name)
    action_unit_files = os.listdir(action_unit_path)
    vid_json = vid_name + ".json"
    stored_label_path = os.path.join(preprocess_label_path, vid_json)
    label_dict = {}
    for au_file in action_unit_files:
        au_file_path = os.path.join(action_unit_path, au_file)
        au_id = int((au_file.split(".")[0])[8:])
        label_dict[au_id] = extract_label_file(au_file_path)
    with open(stored_label_path, "w") as label_outfile:
        json.dump(label_dict, label_outfile)


def form_label_dict(label_path, preprocess_label_path):
    if not os.path.exists(preprocess_label_path):
       os.mak(preprocess_label_path)
    for vid_name in os.listdir(label_path):
        extract_labels(vid_name)

def extract_au_labels(input_folders, au_id):
    labels = []
    vid_inds = []
    frame_inds = []
    for i in range(len(input_folders)):
        in_file = "{folder_name}_au{au_id}.txt".format(input_folders[i], au_id)
        label_dict = extract_label_file(in_file)
        label_list = extract_label_list(in_file)
        vid_inds_curr = [""] * len(label_list) 
        labels.extend(label_list)
        curr_name = extract_filename_no_ext(input_folders[i])
        frame_inds_curr = range(0, len(label_list))
        frame_inds.extend(list(frame_inds_curr))
        for j in range(len(vid_inds_curr)):
            vid_inds_curr[j] = curr_name
        vid_inds.extend(vid_inds_curr)
    return labels, vid_inds, frame_inds
        