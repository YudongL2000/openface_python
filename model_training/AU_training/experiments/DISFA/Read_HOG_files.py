import numpy as np
import struct
import os


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
        f.close()
        return is_valid, feature_vectors



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



def Read_HOG_files(users, hog_data_dir):
    hog_data = np.array([])
    vid_id = []
    valid_inds = np.array([])
    hog_data_files = []
    for i in range(len(users)):
        hog_file = os.path.join(hog_data_dir, "LeftVideo"+users[i]+"_comp.hog")
        hog_data_files.append(hog_file)
    for hog_file in hog_data_files:
        is_valid, feature_vectors = read_hog(hog_file)
        curr_ind_tmp = feature_vectors.shape[0]
        print(is_valid)
        #valid_inds = np.concatenate((valid_inds, is_valid), axis=0)
        vid_id_curr = []
        for j in range(curr_ind_tmp):
            vid_id_curr.append(hog_file)
        is_valid = is_valid[0:curr_ind_tmp]
        curr_data = feature_vectors[0:curr_ind_tmp, :]
        vid_id_curr = vid_id_curr[0:curr_ind_tmp]
        valid_inds = np.concatenate((valid_inds, is_valid), axis=0)
        vid_id += vid_id_curr
        print(curr_data.shape)
        if (hog_data.shape[0] == 0):
            hog_data = curr_data
        else:
            hog_data = np.concatenate((hog_data, curr_data), axis = 0)
        #hog_data += curr_data_small
    hog_data = np.array(hog_data)
    print(hog_data.shape)
    return (hog_data, valid_inds, vid_id)


def Read_HOG_files_dynamic(users, hog_data_dir):
    hog_data = np.array([])
    vid_id = []
    valid_inds = np.array([])
    hog_data_files = []
    for i in range(len(users)):
        hog_file = os.path.join(hog_data_dir, "LeftVideo"+users[i]+"_comp.hog")
        hog_data_files.append(hog_file)
    for hog_file in hog_data_files:
        is_valid, feature_vectors = read_hog(hog_file)
        curr_ind_tmp = feature_vectors.shape[0]
        print(is_valid)
        #valid_inds = np.concatenate((valid_inds, is_valid), axis=0)
        vid_id_curr = []
        for j in range(curr_ind_tmp):
            vid_id_curr.append(hog_file)
        is_valid = is_valid[0:curr_ind_tmp]
        curr_data = feature_vectors[0:curr_ind_tmp, :]
        median = np.median(curr_data[is_valid])
        curr_data = curr_data - median
        vid_id_curr = vid_id_curr[0:curr_ind_tmp]
        valid_inds = np.concatenate((valid_inds, is_valid), axis=0)
        vid_id += vid_id_curr
        print(curr_data.shape)
        if (hog_data.shape[0] == 0):
            hog_data = curr_data
        else:
            hog_data = np.concatenate((hog_data, curr_data), axis = 0)
        #hog_data += curr_data_small
    hog_data = np.array(hog_data)
    print(hog_data.shape)
    return (hog_data, valid_inds, vid_id)
        