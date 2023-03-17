import matplotlib.pyplot as plt
import numpy as np
import json

#3D mappings
def load_json_landmarks_normalization(landmark_path, C=100):
    json_file = open(landmark_path)
    data_dict = json.load(json_file)
    matrix_data = []
    row_ids = []
    list_keys = list(data_dict.keys())
    list_keys.sort()
    for frame_id in list_keys:
        landmark_vectors = np.array(data_dict[frame_id])
        frame_number = int(frame_id.split("_")[-1])
        if (len(landmark_vectors) == 0):
            continue
        else:
            landmark_vectors *= [1,1,-1]
            landmark_vectors *= C
            matrix_data.append(landmark_vectors)
            row_ids.append(frame_number)
    matrix_data = np.array(matrix_data)
    return matrix_data, row_ids