import os
from extract_labels import extract_au_labels
import numpy as np



def create_file_list_single_au_classification(AU_dir, aus):
    AU_dir = 'D:/Databases/DISFA/ActionUnit_Labels/'

    aus = [1,2,4,5,6,9,12,15,17,20,25,26]

    subjects = []
    for filename in os.listdir(AU_dir):
        if filename.startswith("SN"):
            subjects.append(filename)
            
    input_label_dirs = [""] * len(subjects)
    for i in range(subjects):
        input_label_dirs[i] = os.path.join(AU_dir + subjects[i], subjects[i])

    for user in range(len(subjects)):
        testing_label_files = input_label_dirs[user]
        training_label_files = input_label_dirs - testing_label_files
        training_labels_all = np.array([])
        testing_labels_all = np.array([])
        for au in aus:
            training_labels, training_vid_inds_all, training_frame_inds_all = extract_au_labels(training_label_files, au)
            testing_labels, testing_vid_inds_all, testing_frame_inds_all = extract_au_labels(testing_label_files, au)
            
            training_labels_all = np.concatenate((training_labels_all, np.array(training_labels)), axis= 1)
            testing_labels_all = np.concatenate((testing_labels_all, np.array(testing_labels)), axis= 1)
            
        for au_ind in range(len(aus)):
            positive_samples = training_labels_all[:,au_ind] > 0
            active_samples = np.sum(training_labels_all, axis=1) > 10
            negative_samples = np.sum(training_labels_all,axis=1) == 0
            neg_inds = np.where(negative_samples < 0)
            neg_to_use = np.random.shuffle(neg_inds)
            neg_to_use_indexes = 2 * np.sum(positive_samples) - np.sum(active_samples | positive_samples)
            neg_to_use = neg_inds[: neg_to_use_indexes]
            negative_samples[:] = False
            negative_samples[neg_to_use] = True
            
            training_samples = positive_samples | active_samples | negative_samples
            f_train_file_list = open("{0}/{1}_au{2}_filelist_train.txt".format("single_au_class", subjects[user], aus[au_ind]), "w")
            sample_inds_train = np.where(training_samples.flatten() != 0)
            
            for sample_ind in sample_inds_train:
                img_file_l = "{0}/{1}_au{2}_filelist_train.txt".format(training_vid_inds_all[sample_ind], training_frame_inds_all[sample_ind])
                img_file_r = "{0}/{1}_au{2}_filelist_train.txt".format(training_vid_inds_all[sample_ind], training_frame_inds_all[sample_ind])
                
                au_class = training_labels_all[sample_ind, au_ind] > 1
                f_train_file_list.write("{} {}\r\n", img_file_l, au_class)
                f_train_file_list.write("{} {}\r\n", img_file_r, au_class)
            f_train_file_list.close()
            
            
            f_test_file_list = open("{0}/{1}_au{2}_filelist_test.txt".format("single_au_class", subjects[user], aus[au_ind]), "w")
            testing_samples = np.ones(testing_samples.shape[0], 1)
            sample_inds_test = np.where(testing_samples.flatten() != 0)
            for sample_ind in sample_inds_test:
                img_file_l = "{0}/{1}_au{2}_filelist_train.txt".format(testing_vid_inds_all[sample_ind], testing_frame_inds_all[sample_ind])
                img_file_r = "{0}/{1}_au{2}_filelist_train.txt".format(testing_vid_inds_all[sample_ind], testing_frame_inds_all[sample_ind])
                
                au_class = testing_labels_all[sample_ind, au_ind] > 1
                f_test_file_list.write("{} {}\r\n".format(img_file_l, au_class))
                f_test_file_list.write("{} {}\r\n".format(img_file_r, au_class))
            f_test_file_list.close()
        
            
        
        
    