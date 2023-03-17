import os
import json
import csv
import numpy as np


def extract_filename_no_ext(file_path):
    head, tail = os.path.split(file_path)
    filename_no_ext = os.path.splitext(tail)[0]
    return filename_no_ext

def extract_BP4D_labels(BP4D_dir, recs, aus):
    aus_BP4D = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23]
    inds_to_use = []
    for i in range(len(aus)):
        inds_to_use.append(aus_BP4D.index(aus[i]))
    csv_files = []
    for f in os.listdir(BP4D_dir):
        if f.endswith(".csv"):
            csv_files.append(f)
    num_files = len(csv_files)
    labels = [None] * num_files
    valid_ids = [None] * num_files
    vid_ids = np.zeros(num_files, 2)
    filenames = [None] * num_files
    file_id = 0
    for i in range(len(recs)):
        csv_files = []
        for f in os.listdir(os.path.join(BP4D_dir), recs[i]):
            if f.endswith(".csv"):
                csv_files.append(f)
        for j in range(len(csv_files)):
            csv_file = os.path.join(BP4D_dir, csv_files[j])
            csv_filename = extract_filename_no_ext(csv_file)
            filenames[file_id] = filenames
            frame_nums = []
            codes = []
            occlusions = []
            with open(csv_file, "r") as csvfile:
                csv_reader = csv.read(csvfile)
                header = next(csv_reader)
                for row in csv_reader:
                    frame_nums.append(row[0])
                    codes.append(row[1:len(row)-1])
                    occlusions.append(row[-1])
            csvfile.close()
            codes = codes[:, aus_BP4D-1]
            
            occlusions = np.array(occlusions)
            codes = np.array(codes)
            
            valid = (occlusions != 1)
            for s in range(codes.shape[1]):
                valid = valid & codes[: s] != 9
            vid_ids[file_id] = [frame_nums[0], frame_nums[-1]]
            labels[file_id] = codes[:, inds_to_use]
            valid_ids[file_id] = valid
            file_id += 1
    labels = labels[:file_id]
    valid_ids = valid_ids[1:file_id-1]
    vid_ids = vid_ids[1:file_id-1, :]
    filenames = filenames[1:file_id-1]
    return labels, valid_ids, vid_ids, filenames



def extract_BP4D_labels_intensity(BP4D_dir, recs, aus):
    files_all = []
    folder_path = os.path.join(BP4D_dir, "AU{:%02d}".format(aus[0]))
    for output_file in os.listdir(folder_path):
        if output_file.endswith(".csv"):
            files_all.append(os.path.join(folder_path, output_file))
    num_files = len(files_all)
    labels = [None] * num_files
    valid_ids = [None] * num_files
    vid_ids = np.zeros((num_files, 2))
    filenames = [None] * num_files
    file_id = 0
    for r in range(len(recs)):
        files_root = os.path.join(BP4D_dir, "AU{:02d}".format(au))
        files_all = []
        for csv_file in os.listdir(files_root):
            if csv_file.endswith(".csv"):
                files_all.append(csv_file)
        for f in range(len(files_all)):
            for au in aus:
                files_all = []
                files_root = os.path.join(os.path.join(BP4D_dir, "AU{:02d}".format(au)), recs[r])
                for csv_file in os.listdir(files_root):
                    if csv_file.endswith(".csv"):
                        files_all.append(csv_file)
                file = os.path.join(files_root, files_all[f])
                filename = extract_filename_no_ext(filename)
                filenames[file_id] = filename[0:7]
                frame_nums = []
                codes = []
                with open(file, newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    header = next(csv_reader)
                    for row in csv_reader:
                        frame_nums.append(row[0])
                        codes.append(row[1])
                csvfile.close()
                valid = (codes!= 9)
                vid_ids[file_id] = [frame_nums[0], len(frame_nums)]
                if (au == aus[0]):
                    valid_ids[file_id] = valid
                    labels[file_id] = []
                    for i in range(len(codes)):
                        labels[file_id][i] = [codes[i]]
                else:
                    valid_ids[file_id] = valid_ids[file_id] & valid
                    for i in range(len(labels[file_id])):
                        labels[file_id][i].append(codes[i])
        file_id += 1
    labels = labels[1:file_id]
    valid_ids = valid_ids[1:file_id]
    vid_ids = vid_ids[1:file_id, :]
    filenames = filenames[1:file_id]
    return (labels, valid_ids, vid_ids, filenames)
            
        
                    
                
                        
                    
                
            
            
                
            
    