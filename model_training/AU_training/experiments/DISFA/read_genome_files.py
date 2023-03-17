import pandas as pd
import json
import numpy as np
import csv
from scipy.io import loadmat
import os

def genome_file_product(video_data, M, V):
    tab = pd.read_csv(video_data)
    column_names = tab.columns
    table_length = len(tab['frame'])
    valid_frame_indexes = []
    frame_success = []
    local_entries = tab.columns[11:]
    global_entries = tab.columns[6:11]
    result_vec = []
    for i in range(table_length):
        success = tab["success"][i]
        if (int(success) == 1):
            valid_frame_indexes.append(i)
            frame_success.append(True)
        else:
            frame_success.append(False)
        global_vec = []
        feature_vec = []
        for entry in local_entries:
            feature_vec.append(tab[entry][i])
        for entry in global_entries:
            global_vec.append(tab[entry][i])
        feature_vec = np.array(feature_vec)
        global_vec = np.array(global_vec)
        result_vec.append(feature_vec)
    result_vec = np.array(result_vec)
    actual_locations = np.dot(result_vec, V.T)
    actual_valid_locations = actual_locations[frame_success]
    print(actual_valid_locations[0])
    return result_vec, actual_valid_locations

def arr_to_xyz(arr):
    result = []
    for i in range(len(arr)):
        coordinates = arr[i]
        point_counter = (arr.shape[1]) // 3
        points = []
        for j in range(point_counter):
            index_x = j
            index_y = j + point_counter
            index_z = 3+ 2 * point_counter
            (x, y, z) = (coordinates[index_x], coordinates[index_y], coordinates[index_z])
            points.append((x, y, z))
        result.append(points)
    return result

def construct_xyz_from_mat_file(mat_file_path, csv_source_file):
    annots = loadmat(mat_file_path)
    print("V SHAPE")
    print(annots["V"].shape)
    print("M SHAPE")
    print(annots["M"].shape)
    arr = genome_file_product(csv_source_file, annots['M'], annots['V'])
    xyz_coordinates = arr_to_xyz(arr)
    return xyz_coordinates


def read_genome_file(users, hog_data_dir, mat_file_path="../../pca_generation/pdm_68_aligned_wild.mat"):
    geom_data = []
    annots = loadmat(mat_file_path)
    M = annots["M"]
    V = annots["V"]
    for i in range(len(users)):
        genome_file = os.path.join(hog_data_dir, "LeftVideo{}_comp.csv".format(users[i]))
        res, actual_valid_locations = genome_file_product(genome_file, M, V)
        res = np.concatenate((actual_valid_locations, res), axis=1)
        geom_data = np.concatenate((geom_data, res), axis=0)
    return geom_data


def read_genome_file_dynamic(users, hog_data_dir, mat_file_path="../../pca_generation/pdm_68_aligned_wild.mat"):
    geom_data = []
    annots = loadmat(mat_file_path)
    M = annots["M"]
    V = annots["V"]
    for i in range(len(users)):
        genome_file = os.path.join(hog_data_dir, "LeftVideo{}_comp.csv".format(users[i]))
        res, actual_valid_locations = genome_file_product(genome_file, M, V)
        res = np.concatenate((actual_valid_locations, res), axis=1)
        res = res - np.median(res)
        geom_data = np.concatenate((geom_data, res), axis=0)
    return geom_data
            
            
            
            