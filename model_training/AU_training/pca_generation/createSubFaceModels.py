import numpy as np
import torch
import os
import csv
import sklearn
from sklearn import PCA
import json
from json import JSONEncoder
from Read_HOG_files_small import read_hog, Read_HOG_files_small

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def create_facial_model(face_processed_dir, dataset_name):
    hog_dir = os.path.join(face_processed_dir, dataset_name)
    hog_files = []
    for f_name in os.listdir(hog_dir):
        if (f_name.endswith(".hog")):
            hog_path = os.path.join(hog_dir, f_name)
            hog_files.append(hog_path)
    appearance_data_tmp, valid_inds_tmp, vid_ids_train_tmp = Read_HOG_files_small(hog_files, hog_dir)
    appearance_data_tmp = appearance_data_tmp[valid_inds_tmp,:]
    vid_ids_train_tmp = vid_ids_train_tmp[valid_inds_tmp,:]
    appearance_data = np.concatenate((appearance_data, appearance_data_tmp), axis=0)
    vid_ids_train = np.concatenate((vid_ids_train, vid_ids_train_tmp), axis=0)
    means_norm = np.mean(appearance_data)
    stds_norm = np.std(appearance_data)
    normed_data = (appearance_data - means_norm) / stds_norm
    pca_model = PCA()
    pca_model.fit(normed_data)
    principal_components = pca_model.components
    score = pca_model.transform(normed_data)
    singular_values = pca_model.singular_values
    store_rigid(principal_components, singular_values, means_norm, stds_norm)
    store_lower(normed_data, means_norm, stds_norm)
    store_upper(normed_data, means_norm, stds_norm)


def store_rigid(principal_components, singular_values, means_norm, stds_norm):
    total_sum = np.sum(np.square(singular_values))
    count = len(singular_values)
    eigenvalue_tracer = 0.0
    for i in range(len(singular_values)):
        eigenvalue_tracer += (singular_values[i])**2
        if (eigenvalue_tracer >= 0.95 * total_sum):
            count = i + 1
            break
    principal_components = principal_components[:count]
    json_dict = {}
    json_dict["PC"] = principal_components
    json_dict["means_norm"] = means_norm
    json_dict["stds_norm"] = stds_norm
    with open("generic_face_rigid.json", "w") as write_file:
        json.dump(json_dict, write_file, cls=NumpyArrayEncoder)

def store_lower(normed_data, means_norm, stds_norm):
    normed_data_lower_face = normed_data
    normed_data_lower_face[:, 0:5*12*31] = 0
    pca_model = PCA()
    pca_model.fit(normed_data_lower_face)
    principal_components = pca_model.components
    score = pca_model.transform(normed_data)
    singular_values = pca_model.singular_values
    total_sum = np.sum(np.square(singular_values))
    count = len(singular_values)
    eigenvalue_tracer = 0.0
    for i in range(len(singular_values)):
        eigenvalue_tracer += (singular_values[i])**2
        if (eigenvalue_tracer >= 0.98 * total_sum):
            count = i + 1
            break
    principal_components = principal_components[:count]
    json_dict = {}
    json_dict["PC"] = principal_components
    json_dict["means_norm"] = means_norm
    json_dict["stds_norm"] = stds_norm
    with open("generic_face_lower.json", "w") as write_file:
        json.dump(json_dict, write_file, cls=NumpyArrayEncoder)


def store_upper(normed_data, means_norm, stds_norm):
    normed_data_upper_face = normed_data
    normed_data_upper_face[:, 5*12*31:] = 0
    pca_model = PCA()
    pca_model.fit(normed_data_upper_face)
    principal_components = pca_model.components
    score = pca_model.transform(normed_data)
    singular_values = pca_model.singular_values
    total_sum = np.sum(np.square(singular_values))
    count = len(singular_values)
    eigenvalue_tracer = 0.0
    for i in range(len(singular_values)):
        eigenvalue_tracer += (singular_values[i])**2
        if (eigenvalue_tracer >= 0.98 * total_sum):
            count = i + 1
            break
    principal_components = principal_components[:count]
    json_dict = {}
    json_dict["PC"] = principal_components
    json_dict["means_norm"] = means_norm
    json_dict["stds_norm"] = stds_norm
    with open("generic_face_upper.json", "w") as write_file:
        json.dump(json_dict, write_file, cls=NumpyArrayEncoder)
            
    