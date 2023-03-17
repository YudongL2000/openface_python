import os
import numpy as np

def extract_au_labels(input_folders, au_id):
    labels = []
    for i in range(len(input_folders)):
        in_file = "{}_au{}.txt".format(input_folders[i], au_id)
        A = []
        with open(in_file, "r") as txt_reader:
            file_lines = txt_reader.readlines()
            label_dict = {}
            for line in file_lines:
                row_id = int(line.split(",")[0])
                au_intensity = int(line.split(",")[1])
                label_dict[row_id] = au_intensity
                A.append(au_intensity)
        labels.append(A)
    labels = np.array(labels)
    return labels