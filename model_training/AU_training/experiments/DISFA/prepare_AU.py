import os
from read_genome_files import genome_file_product, read_genome_file
from scipy.io import loadmat
import csv
from Read_HOG_files import read_hog, Read_HOG_files, Read_HOG_files_dynamic
from utils import extract_au_labels
import numpy as np
import json

def prepare_data_generic_json(train_users, test_users, au_train, au_test, rest_aus,  root, features_dir):
    input_train_label_files = [None] * len(train_users)
    input_test_label_files = [None] * len(test_users)
    for i in range(len(train_users)):
        input_train_label_files[i] = os.path.join(
            os.path.join(
                os.path.join(root, "ActionUnit_Labels"), 
                train_users[i]), 
            train_users[i])
    for i in range(len(test_users)):
        input_test_label_files[i] = os.path.join(
        os.path.join(
            os.path.join(root, "ActionUnit_Labels"), 
            test_users[i]), 
        test_users[i])
    train_genome_data = read_genome_file(train_users, features_dir)
    test_genome_data = read_genome_file(test_users, features_dir)
    train_data, tracked_inds_hog, vid_ids_train = Read_HOG_files(train_users, features_dir)
    test_data, success_test, vid_ids_test = Read_HOG_files(test_users, features_dir)
    
    train_data = np.concatenate((train_data, train_genome_data), axis=1)
    raw_test= np.concatenate((test_data, train_genome_data), axis=1)
    
    labels_train = extract_au_labels(input_train_label_files, au_train)
    labels_test = extract_au_labels(input_test_label_files, au_test)
    labels_other = np.zeros(labels_train.shape[0], len(rest_aus))
    if(len(input_train_label_files) > 0):
        for i in range(len(rest_aus)):
            labels_other[:, i] = extract_au_labels(input_train_label_files, rest_aus[i])
        reduced_inds = np.zeros(len(labels_train), dtype=bool)
        reduced_inds[labels_train>=0] = True
        
        pos_count = np.sum(labels_train > 0)
        neg_count = np.sum(labels_train == 0)
        
        num_other = np.floor(pos_count / (labels_other.shape[1]))
        inds_all = list(range(0, len(labels_train)))
        inds_all = np.array(inds_all)
        for i in range(labels_other.shape[1]+1):
            if(i >= labels_other.shape[1]):
                inds_choices = np.sum(labels_other,1)==0 & (labels_train == 0)
                inds_other = inds_all[inds_choices]
                num_other_i = np.min(len(inds_other), pos_count - np.sum(labels_train[reduced_inds,:]==0))   
            else:
                inds_choices = labels_other[:, i] & (labels_train == 0)
                inds_other = inds_all[inds_choices]      
                num_other_i = np.min(len(inds_other), num_other)     
            inds_choices = (np.linspace(1, len(inds_other), num_other_i)).round() 
            inds_other_to_keep = inds_other[inds_choices]
            reduced_inds[inds_other_to_keep] = True
        tracked_index = tracked_inds_hog == 0
        reduced_inds[tracked_index] = False
        labels_train = labels_train[reduced_inds]
        train_data = train_data[reduced_inds, :]
    genome_size = max(train_genome_data.shape[1], test_genome_data.shape[1])
    
    if (au_train < 8 or au_train == 43 or au_train == 45):
        pca_file = "../../pca_generation/generic_face_upper.json"
        annots=json.load(open(pca_file, "r"))
    elif (au_train > 9):
        pca_file = "../../pca_generation/generic_face_lower.json"
        annots=json.load(open(pca_file, "r"))
    elif (au_train == 9):
        pca_file = "../../pca_generation/generic_face_rigid.json"
        annots=json.load(open(pca_file, "r"))
    PC = annots["PC"]
    PC_n = np.zeros(len(PC)+ genome_size)
    PC_n[:PC.shape[0], 1:PC.shape[1]] = PC
    PC_n[PC.shape[0]:, PC.shape[1]:] = PC
    PC = PC_n
    means_norm = annots["means_norm"]
    stds_norm = annots["stds_norm"]
    means_norm = np.concatenate((means_norm, np.zeros([1, genome_size])), axis = 1)
    stds_norm = np.concatenate((stds_norm, np.zeros([1, genome_size])), axis = 1)
    
    data_test = (raw_test - means_norm)/stds_norm
    data_test = data_test * PC
    
    if (len(train_data) > 0):
        data_train = (train_data - means_norm)/stds_norm
        data_train = data_train * PC
    else:
        data_train = []
    return (data_train, labels_train, data_test, 
            labels_test, raw_test, PC, 
            means_norm, stds_norm, vid_ids_test, 
            success_test)
    
    
    
def prepare_data_generic_dynamic_json(train_users, test_users, au_train, au_test, rest_aus,  root, features_dir):
    input_train_label_files = [None] * len(train_users)
    input_test_label_files = [None] * len(test_users)
    for i in range(len(train_users)):
        input_train_label_files[i] = os.path.join(
            os.path.join(
                os.path.join(root, "ActionUnit_Labels"), 
                train_users[i]), 
            train_users[i])
    for i in range(len(test_users)):
        input_test_label_files[i] = os.path.join(
        os.path.join(
            os.path.join(root, "ActionUnit_Labels"), 
            test_users[i]), 
        test_users[i])
    train_genome_data = read_genome_file(train_users, features_dir)
    test_genome_data = read_genome_file(test_users, features_dir)
    train_data, tracked_inds_hog, vid_ids_train = Read_HOG_files_dynamic(train_users, features_dir)
    test_data, success_test, vid_ids_test = Read_HOG_files_dynamic(test_users, features_dir)
    
    train_data = np.concatenate((train_data, train_genome_data), axis=1)
    raw_test= np.concatenate((test_data, train_genome_data), axis=1)
    
    labels_train = extract_au_labels(input_train_label_files, au_train)
    labels_test = extract_au_labels(input_test_label_files, au_test)
    labels_other = np.zeros(labels_train.shape[0], len(rest_aus))
    if(len(input_train_label_files) > 0):
        for i in range(len(rest_aus)):
            labels_other[:, i] = extract_au_labels(input_train_label_files, rest_aus[i])
        reduced_inds = np.zeros(len(labels_train), dtype=bool)
        reduced_inds[labels_train>=0] = True
        
        pos_count = np.sum(labels_train > 0)
        neg_count = np.sum(labels_train == 0)
        
        num_other = np.floor(pos_count / (labels_other.shape[1]))
        inds_all = list(range(0, len(labels_train)))
        inds_all = np.array(inds_all)
        for i in range(labels_other.shape[1]+1):
            if(i >= labels_other.shape[1]):
                inds_choices = np.sum(labels_other,1)==0 & (labels_train == 0)
                inds_other = inds_all[inds_choices]
                num_other_i = np.min(len(inds_other), pos_count - np.sum(labels_train[reduced_inds,:]==0))   
            else:
                inds_choices = labels_other[:, i] & (labels_train == 0)
                inds_other = inds_all[inds_choices]      
                num_other_i = np.min(len(inds_other), num_other)     
            inds_choices = (np.linspace(1, len(inds_other), num_other_i)).round() 
            inds_other_to_keep = inds_other[inds_choices]
            reduced_inds[inds_other_to_keep] = True
        tracked_index = tracked_inds_hog == 0
        reduced_inds[tracked_index] = False
        labels_train = labels_train[reduced_inds]
        train_data = train_data[reduced_inds, :]
    genome_size = max(train_genome_data.shape[1], test_genome_data.shape[1])
    
    if (au_train < 8 or au_train == 43 or au_train == 45):
        pca_file = "../../pca_generation/generic_face_upper.json"
        annots=json.load(open(pca_file, "r"))
    elif (au_train > 9):
        pca_file = "../../pca_generation/generic_face_lower.json"
        annots=json.load(open(pca_file, "r"))
    elif (au_train == 9):
        pca_file = "../../pca_generation/generic_face_rigid.json"
        annots=json.load(open(pca_file, "r"))
    PC = annots["PC"]
    PC_n = np.zeros(len(PC)+ genome_size)
    PC_n[:PC.shape[0], 1:PC.shape[1]] = PC
    PC_n[PC.shape[0]:, PC.shape[1]:] = PC
    PC = PC_n
    means_norm = annots["means_norm"]
    stds_norm = annots["stds_norm"]
    means_norm = np.concatenate((means_norm, np.zeros([1, genome_size])), axis = 1)
    stds_norm = np.concatenate((stds_norm, np.zeros([1, genome_size])), axis = 1)
    
    data_test = (raw_test - means_norm)/stds_norm
    data_test = data_test * PC
    
    if (len(train_data) > 0):
        data_train = (train_data - means_norm)/stds_norm
        data_train = data_train * PC
    else:
        data_train = []
    return (data_train, labels_train, data_test, 
            labels_test, raw_test, PC, 
            means_norm, stds_norm, vid_ids_test, 
            success_test)
    