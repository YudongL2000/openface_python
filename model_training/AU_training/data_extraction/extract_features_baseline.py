import os
import subprocess

def extract_features_baseline_DISFA(feature_exe, input_dir, output_dir):
    print("extracting features from left videos")
    DISFA_location_Left = os.path.join(input_dir, "Video_LeftCamera")
    left_videos = []
    for video in os.listdir(DISFA_location_Left):
        if video.endswith(".avi"):
            left_videos.append(os.path.join(DISFA_location_Left, video))
            input_file = os.path.join(DISFA_location_Left, video)
            exe_call = "./" + feature_exe
            subprocess.run([exe_call, "-f", input_file, "-out_dir", output_dir, "-hogalign", "-pdmparams"])
    print("extracting features from right videos")
    DISFA_location_Right = os.path.join(input_dir, "Video_RightCamera")
    right_videos = []
    for video in os.listdir(DISFA_location_Right):
        if video.endswith(".avi"):
            right_videos.append(os.path.join(DISFA_location_Right, video))
            input_file = os.path.join(DISFA_location_Right, video)
            exe_call = "./" + feature_exe
            subprocess.run([exe_call, "-f", input_file, "-out_dir", output_dir, "-hogalign", "-pdmparams"])

    
    
    
