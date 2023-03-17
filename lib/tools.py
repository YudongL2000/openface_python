import os
import json
import cv2
import mediapipe as mp
import os
import numpy as np
import matplotlib.pyplot as plt

def load_image_paths(name_dir, facial_raw_images_path):
    rigid_faces = os.path.join(facial_raw_images_path, name_dir)
    facial_images = os.listdir(rigid_faces)
    faces = []
    for img in facial_images:
        pth = os.path.join(rigid_faces, img)
        faces.append(pth)
    return faces



def load_all_images(action_unit_label_path, facial_raw_images_path):
    Left_images = {}
    Right_images = {}
    action_unit_folders = os.listdir(action_unit_label_path)
    print(action_unit_folders)
    for vid_name in action_unit_folders:
        #print(vid_name)
        if (vid_name.startswith("SN") == False):
            continue
    left_images_folder = "LeftVideo" + vid_name + "_comp"
    right_images_folder = "RightVideo" + vid_name + "_comp"
    left_images_path = os.path.join(facial_raw_images_path, left_images_folder)
    right_images_path = os.path.join(facial_raw_images_path, right_images_folder)
    faces_left = load_image_paths(left_images_folder)
    faces_right = load_image_paths(right_images_folder)
    Left_images[vid_name] = faces_left
    Right_images[vid_name] = faces_right
    return Left_images, Right_images


def landmark_detection(face_mesh, image_path, save_path, display= True):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    height, width, _ = image.shape
    landmarks = []
    if not result.multi_face_landmarks:
        return []
    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
            #print(width, height)
            #print(pt1.x, pt1.y, pt1.z)
            landmarks.append([pt1.x, pt1.y, pt1.z])
            cv2.circle(image, (x, y), 1, (100, 100, 0), -1)
    landmarks = np.array(landmarks)
    cv2.imwrite(save_path, image)
    if display:
        cv2.imshow(image)
    #display_3D_points(landmarks)
    return landmarks

def store_landmarks(image_key, store_landmark_image_path, store_landmark_jsons, Left_images, Right_images):
    left_images_folder = "LeftVideo" + image_key + "_comp"
    right_images_folder = "RightVideo" + image_key + "_comp"
    json_folder = os.path.join(store_landmark_jsons, image_key)
    jsonExist = os.path.exists(json_folder)
    if jsonExist == False:
        os.mkdir(json_folder)
    processed_image_folder_left = os.path.join(store_landmark_image_path, left_images_folder)
    processed_image_folder_right = os.path.join(store_landmark_image_path, right_images_folder)
    processed_left_exist = os.path.exists(processed_image_folder_left)
    processed_right_exist = os.path.exists(processed_image_folder_right)
    if (processed_left_exist == False):
        os.mkdir(processed_image_folder_left)
    if (processed_right_exist == False):
        os.mkdir(processed_image_folder_right)
    left_json_name = "Left_" + image_key + ".json"
    left_json_path = os.path.join(json_folder, left_json_name)
    right_json_name = "Right_" + image_key + ".json"
    right_json_path = os.path.join(json_folder, right_json_name)
    print("storing left facial landmark in json path", left_json_path)
    print("storing right facial landmark in json path", right_json_path)
    left_result = {}
    right_result = {}
    for i in range(len(Left_images[image_key])):
        left_image_path = Left_images[image_key][i]
        left_image_name = left_image_path.split("/")[-1]
        left_image_name_no_ext = left_image_name.split(".")[0]
        right_image_path = Right_images[image_key][i]
        right_image_name = left_image_name
        right_image_name_no_ext = left_image_name_no_ext

    display_images = False
    if (i % 1000 == 0):
      display_images = True
    left_save_path = os.path.join(processed_image_folder_left, left_image_name)
    right_save_path = os.path.join(processed_image_folder_right, right_image_name)
    left_landmarks = landmark_detection(left_image_path, left_save_path, display_images)
    right_landmarks = landmark_detection(right_image_path, right_save_path, display_images)

    if (i % 1000 == 0):
      display_3D_points(left_landmarks)
      display_3D_points(right_landmarks)

    left_result[left_image_name_no_ext] = left_landmarks
    right_result[right_image_name_no_ext] = right_landmarks
    with open(left_json_path, "w") as left_outfile:
        json.dump(left_result, left_outfile)
    with open(right_json_path, "w") as right_outfile:
        json.dump(right_result, right_outfile)


def create_face_mesh():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5)
    return face_mesh

def display_3D_points(landmark_lists):
    X_list = []
    Y_list = []
    Z_list = []
    for pt in landmark_lists:
        #print(pt)
        x = pt[0]
        y = pt[1]
        z = pt[2]
        X_list.append(x)
        Y_list.append(y)
        Z_list.append(z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_list, Y_list, Z_list)
    plt.show()

