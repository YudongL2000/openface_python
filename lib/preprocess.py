import os
import subprocess
from os.path import exists, join, basename, splitext
from IPython.display import YouTubeVideo
import struct
import numpy as np

#data loading for each label from video files
def extract_au_labels(input_folders, au_id):
    labels = []
    vid_inds = []
    frame_inds = []
    for i in range(len(input_folders)):
        input_foldername = input_folders[i]
        input_foldername_last = input_foldername.split("/")[-1]
        input_file = input_foldername_last + "_au"+ str(au_id)+".txt"
        input_file_path = os.path.join(input_foldername, input_file)
        res_table = []
        file_reader = open(input_file_path, "r")
        for line in file_reader.readlines():
            """
            if (len(line.split(",")) < 2):
            print(input_file_path)
            print(line)
            break
            """
            idx = line.split(",")[0]
            label = line.split(",")[1]
            res_table.append([float(idx), float(label)])
        res_table = np.array(res_table)
        vid_inds_curr = [None] * len(res_table);
        labels = np.concatenate((labels, res_table[:,1]));
        frame_inds_curr = list(range(0, len(res_table)));
        frame_inds = np.concatenate((frame_inds, frame_inds_curr))
        for i in range(len(res_table)):
            vid_inds_curr[i] = input_foldername_last
        vid_inds += vid_inds_curr
    return (labels, vid_inds, frame_inds)

#preprocess DISFA
def preprocess_DISFA_baseline(input_dir):
    #find the location of DISFA data
    hog_data_dir = os.path.join(input_dir, "hog_aligned_rigid")
    users = ['SN001','SN002','SN003','SN004','SN005','SN006','SN007','SN008','SN009', 'SN010','SN011','SN012','SN016','SN017','SN018','SN021','SN023','SN024','SN025',
             'SN026','SN027','SN028','SN029','SN030','SN031','SN032','SN033']
    AU_dir = os.path.join(input_dir, "ActionUnit_Labels");
    aus = [1,2,4,5,6,9,12,15,17,20,25,26];
    names = []
    subjects = []
    for dir_txt in os.listdir(AU_dir):
        print(dir_txt)
        if dir_txt.startswith('SN'):
            subjects.append(dir_txt)
            names.append(dir_txt)
    for i in range(len(subjects)):
        subject = subjects[i]
        new_subject = os.path.join(AU_dir, subject)
        subjects[i] = new_subject
    input_labels = {}
    for i in range(len(subjects)):
        subject_path = subjects[i]
        name = names[i]
        input_labels[name] = []
        for file in os.listdir(subject_path):
            input_labels[name].append(os.path.join(subject_path, file))
            
    print(testing_labels_dirs)
    print(subjects)
    for user_id in range(len(names)):
        subject_name = names[user_id]
        training_labels_dirs = subjects[:user_id] + subjects[user_id+1:]
        testing_labels_dirs = [subjects[user_id]]
        training_labels_all = []
        testing_labels_all = []
        for au in aus:
            (training_labels, training_video_indexes, training_frame_indexes) = extract_au_labels(training_labels_dirs, au)
            (testing_labels, testing_video_indexes, testing_frame_indexes) = extract_au_labels(testing_labels_dirs, au)
            training_labels = np.expand_dims(training_labels, axis = 1)
            testing_labels = np.expand_dims(testing_labels, axis = 1)
            training_labels_all=np.concatenate((training_labels_all, training_labels), 1)
            testing_labels_all=np.concatenate((testing_labels_all, testing_labels), 1)

        for au_id in range(len(aus)):
            au_val = aus[au_id]
            (_, training_video_indexes, training_frame_indexes) = extract_au_labels(training_labels_dirs, au_val)
            (_, testing_video_indexes, testing_frame_indexes) = extract_au_labels(testing_labels_dirs, au_val)
            training_labels_local = training_labels_all[:, au_id]
            
            positive_samples_mask = training_labels_local> 0
            #positive_samples = training_labels_local[positive_mask]
            
            active_samples_mask = (np.sum(training_labels_all,axis=0) > 10)
            #active_samples = np.sum(training_labels_all,axis=1)[active_samples_mask]
            
            negative_samples_mask = (np.sum(training_labels_all,axis=0) == 0)
            #negative_samples = sum(training_labels_all,2)[negative_samples_mask]


            neg_inds = np.nonzero(negative_samples_mask)
            neg_to_use = np.random.permutation(len(neg_inds))
            selected_neg_inds_indexes = neg_to_use[0:(2*sum(positive_samples_mask) - sum(numpy.ma.mask_or(active_samples_mask, positive_samples_mask)))]
            neg_to_use = neg_inds[selected_neg_inds_indexes]

            negative_samples = np.copy(negative_samples_mask)
            negative_samples = np.zeros(len(negative_samples_mask))
            negative_samples[neg_to_use] = true
                

            training_samples = np.ma.mask_or(negative_samples, np.ma.mask_or(positive_samples_mask, active_samples_mask))
            training_file_list_name = 'single_au_class/%s_au%02d_filelist_train.txt'%(subject_name, aus[au_id])
                    
            f_train_file_list = open(training_file_list_name, 'w')
            sample_inds_train = np.nonzero(training_samples)
            for sample_ind in sample_inds_train:   
                img_file_l = 'LeftVideo%s_comp/frame_det_%06d.png' %(training_video_indexes[sample_ind], training_frame_indexes[sample_ind])
                img_file_r = 'RightVideo%s_comp/frame_det_%06d.png'%(training_video_indexes[sample_ind], training_frame_indexes[sample_ind])
                au_class = training_labels_all[sample_ind, au_id] > 1
                f_train_file_list.write('%s %d\r\n'%(img_file_l, au_class))
                f_train_file_list.write('%s %d\r\n'%(img_file_r, au_class))
            f_train_file_list.close()

            test_file_list_name = 'single_au_class/%s_au%02d_filelist_test.txt'%(subject_name, aus[au_id])
            f_test_file_list = open(test_file_list_name, 'w')
            testing_samples = np.ones(testing_labels_all.shape[0], 1)
            sample_inds_test = np.nonzero(testing_samples)
            for sample_ind in sample_inds_test:
                img_file_l = 'LeftVideo%s_comp/frame_det_%06d.png'%(testing_vid_inds_all[sample_ind], testing_frame_inds_all[sample_ind])
                img_file_r = 'RightVideo%s_comp/frame_det_%06d.png'%(testing_vid_inds_all[sample_ind], testing_frame_inds_all[sample_ind])
                au_class = testing_labels_all[sample_ind, au_ind] > 1
                f_test_file_list.write('%s %d\r\n'%(img_file_l, au_class))
                f_test_file_list.write('%s %d\r\n'%(img_file_r, au_class))
            f_test_file_list.close()



def read_hog(filename, batch_size=5000):
    all_feature_vectors = []
    with open(filename, "rb") as f:
        num_cols, = struct.unpack("i", f.read(4))
        num_rows, = struct.unpack("i", f.read(4))
        num_channels, = struct.unpack("i", f.read(4))
        # The first four bytes encode a boolean value whether the frame is valid
        num_features = 1 + num_rows * num_cols * num_channels
        feature_vector = struct.unpack("{}f".format(num_features), f.read(num_features * 4))
        feature_vector = np.array(feature_vector).reshape((1, num_features))
        all_feature_vectors.append(feature_vector)

        # Every frame contains a header of four float values: num_cols, num_rows, num_channels, is_valid
        num_floats_per_feature_vector = 4 + num_rows * num_cols * num_channels
        # Read in batches of given batch_size
        num_floats_to_read = num_floats_per_feature_vector * batch_size
        # Multiply by 4 because of float32
        num_bytes_to_read = num_floats_to_read * 4

        while True:
            bytes = f.read(num_bytes_to_read)
            # For comparison how many bytes were actually read
            num_bytes_read = len(bytes)
            assert num_bytes_read % 4 == 0, "Number of bytes read does not match with float size"
            num_floats_read = num_bytes_read // 4
            assert num_floats_read % num_floats_per_feature_vector == 0, "Number of bytes read does not match with feature vector size"
            num_feature_vectors_read = num_floats_read // num_floats_per_feature_vector
            feature_vectors = struct.unpack("{}f".format(num_floats_read), bytes)
            # Convert to array
            feature_vectors = np.array(feature_vectors).reshape((num_feature_vectors_read, num_floats_per_feature_vector))
            # Discard the first three values in each row (num_cols, num_rows, num_channels)
            feature_vectors = feature_vectors[:, 3:]
            # Append to list of all feature vectors that have been read so far
            all_feature_vectors.append(feature_vectors)
            if num_bytes_read < num_bytes_to_read:
                break
        # Concatenate batches
        all_feature_vectors = np.concatenate(all_feature_vectors, axis=0)
        # Split into is-valid and feature vectors
        is_valid = all_feature_vectors[:, 0]
        feature_vectors = all_feature_vectors[:, 1:]
        return is_valid, feature_vectors


#Read in Hog Data
def Read_HOG_files_small(hog_files):
    hog_data = np.array([])
    vid_id = []
    valid_inds = np.array([])
    feats_filled = 0
    num_frames_per_vid = 20
    for i in range(len(hog_files)):
        hog_file = hog_files[i]
        is_valid, feature_vectors = read_hog(hog_file)
        curr_ind_tmp = feature_vectors.shape[0]
        print(is_valid)
        #valid_inds = np.concatenate((valid_inds, is_valid), axis=0)
        vid_id_curr = []
        for j in range(curr_ind_tmp):
            vid_id_curr.append(hog_file)
        #vid_id = np.concatenate((vid_id, vid_id_curr), axis = 0)
        #increment: increase number of frames each time
        increment = curr_ind_tmp // num_frames_per_vid + 1
        if (increment == 0):
            increment = 1
        print(feature_vectors.shape)
        assert(is_valid.shape[0] == feature_vectors.shape[0])
        assert(len(vid_id_curr) == is_valid.shape[0])
        is_valid = is_valid[0:curr_ind_tmp:increment]
        curr_data_small = feature_vectors[0:curr_ind_tmp:increment, :]
        vid_id_curr = vid_id_curr[0:curr_ind_tmp:increment]
        valid_inds = np.concatenate((valid_inds, is_valid), axis=0)
        vid_id += vid_id_curr
        print(curr_data_small.shape)
        if (hog_data.shape[0] == 0):
            hog_data = curr_data_small
        else:
            hog_data = np.concatenate((hog_data, curr_data_small), axis = 0)
      #hog_data += curr_data_small
    hog_data = np.array(hog_data)
    print(hog_data.shape)
    return (hog_data, valid_inds, vid_id)


