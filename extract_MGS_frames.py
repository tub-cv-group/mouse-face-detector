"""
    Part of Mouse Face Detector

    Author: Niek Andresen
    Contact: n.andresen@tu-berlin.de, andresen.niek@gmail.com
    Affiliation: TU Berlin, Computer Vision & Remote Sensing
    License: MIT

    Part of the publication:
    "Automatic pain face analysis in mice: Applied to a varied dataset with non-standardized conditions"
    DOI: not yet published

    Created: 2021-03-01
    Last Modified: 2026-01-30
    
    Takes a video and extracts some frames from it, that are well suited for MGS
    judgment. Looks at blur and the visibility of mouse facial features.
    
    Steps:
    - Detect facial features with a trained DLC model
    - Find suitable frames by balancing the number of detected features and the likelihoods of the detections
    - Determine the face bounding boxes
    - Extract the requested frames from the video file
    - Crop the faces and save the cropped images
    - Rank the results by blurriness and store the order in a 'infofile.txt' file next to the crops

    The application of the trained model to find the facial features is best done on GPU (e.g. with the DLC docker).
    The rest is done on CPU.
    Experience from our dataset: All frames of the video are extracted at one point requiring at least 12GB (hard
    drive) free. The whole script takes about an hour for one of the three minute videos of our dataset on an GTX
    1060 and about half that time on a Quadro RTX 4000.
"""

import cv2
import os
from shutil import rmtree
import argparse
from video_to_frames import video_to_frames as get_frames
from detect_blur import detect_blur_fft
import face_detector.detections_to_face as fd
from crop_faces import crop
import numpy as np

import matplotlib.pyplot as plt
from skimage.io import imread, imsave

def show_img(img, win_name='image'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 640)#960, 540)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)

def get_video_shape(vid):
    vcap = cv2.VideoCapture(vid)
    if vcap.isOpened(): 
        width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return (int(width), int(height))

def get_ranks(arr):
    temp = np.array(arr).argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks

def main(vpath, model, num, lhth, rmframes=False):
    # call DLC to detect facial features in the video
    # For a video in our dataset, takes about 40 minutes on GTX 1060, 20 minutes on Quadro RTX 4000 (if model was given, otherwise it is assumed it has been applied before and the results are searched)
    if model is None: # if no DLC model was given, it could be, that the DLC results are already there. That is checked here.
        dlc_feats = None
        for f in os.listdir(os.path.dirname(vpath)):
            if f.endswith('_bx.h5') and os.path.basename(vpath).rsplit('.', 1)[0] in f:
                if dlc_feats is not None:
                    print("Error: Found several matching files in {}, that could be the detected features of the given video. Exiting.".format(os.path.dirname(vid)))
                    exit(1)
                dlc_feats = os.path.join(os.path.dirname(vpath), f)
        if dlc_feats is None:
            print("Error: Found no detected features in {}. Give -m <pathToTrainedModelConfig> and try again. Exiting.".format(os.path.dirname(vpath)))
            exit(1)
    else: # A DLC model was given. Use it to detect the facial features (better use GPU computer).
        import call_DLC as applyDLC
        print("Applying DeepLabCut.")
        dlc_feats = applyDLC.main(vpath, model)
    if dlc_feats is None or not os.path.exists(dlc_feats): # nothing was detected or the video file is corrupted
        with open(vpath.rsplit('.',1)[0] + "_infofile.txt", 'w') as f:
            f.write(f"No faces for MGS rating could be extracted from video {vpath}. Check the video, maybe it doesn't show mice or it is corrupted.\n")
        return

    # get face detections from DLC detections (for our data it takes about 7 minutes (num=1000))
    print("Computing Face Bounding Boxes.")
    shape = get_video_shape(vpath)
    faces = fd.main(dlc_feats, num, lhth, shape)
    frame_indices = faces[0]
    
    # extract frames, if it hasn't been done yet (for our data it takes less than 10 minutes)
    print("Extracting frames from the video.")
    frames_folder = vpath.rsplit('.',1)[0]+"_frames"
    if not os.path.exists(frames_folder):
        get_frames(vpath, frames_folder, frames_to_extract=frame_indices)
    else:
        print(f"Warning: Frames folder exists. This means no new frames are extracted, which could cause an error if other frames are needed.\nIn case of that error, delete the frames folder {frames_folder} and try again.")

    print("Cropping faces out, computing blur and writing logfile.")

    # crop faces out
    result_folder = vpath.rsplit('.',1)[0]+"_MGSframes"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        for idx, bbox in zip(faces[0], faces[1]):
            fname = f"{idx:05d}.png"
            crop(os.path.join(frames_folder, fname), bbox, os.path.join(result_folder, fname))
    if rmframes: rmtree(frames_folder)

    # blur detection
    # Uses two methods. Ranks the results by their performance in each method and then orders them by their average rank.
    laps = []
    fftbs = []
    for idx, face in zip(faces[0], faces[1]):
        img = cv2.imread(os.path.join(result_folder, f"{idx:05d}.png"), cv2.IMREAD_GRAYSCALE)
        laps.append(cv2.Laplacian(img, cv2.CV_64F).var())
        fftbs.append(detect_blur_fft(img)[0])
    sorted_by_avg_rank = np.argsort([(v[0]+v[1])/2 for v in zip(get_ranks(laps), get_ranks(fftbs))])
    faces = ([faces[0][i] for i in sorted_by_avg_rank], [faces[1][i] for i in sorted_by_avg_rank])
    
    # write blurriness ranking/log file
    with open(os.path.join(result_folder, 'infofile.txt'), 'w') as f:
        f.write(f"Frames for MGS rating\nextracted from video {vpath}.\nSorted by blurriness (best quality to worst):\n")
        for idx in faces[0]:
            f.write(f"{idx:05d}.png\n")
        f.write('\n')
        f.write("Params used:\n")
        f.write(f"video: {vpath}\ntrained model: {model}\nnumber of faces requested: {num}\nlikelihood threshold: {lhth}\nremove frames afterwards: {rmframes}\n")

if __name__=="__main__":
    p = argparse.ArgumentParser(description="Extracts frames suitable for MGS analysis from a video. For this a trained DLC model is applied to find the facial features, then face bounding boxes are found.")
    p.add_argument('vid', help="The video with the mouse or mice.")
    p.add_argument("-m", "--model", help="The config.yaml file of the trained DLC model. Default: It is assumed, that it has already been applied and the script looks for the results next to the video", default=None)
    p.add_argument("-n", "--num_candidates", help="The maximum number of face detections to return for each mouse. Will be less than this, if there were fewer frames. Experience shows, that when one chooses number bigger than 20 or so, that there will be quite a few suboptimal detections with nose cropped off etc. Default: 15", default=15, type=int)
    p.add_argument("-l", "--likelihood_thresh", help="The threshold under which a detection is not used. Default: 0.8", default=.8, type=float)
    p.add_argument('-kf', '--keep_frames', help="If this is set, the folder with all of the frames in the video, that is created for this program, is kept and not deleted. This does not affect the output folder with the MGS-suitable images.", action="store_true")
    args = p.parse_args()
    main(args.vid, args.model, args.num_candidates, args.likelihood_thresh, not args.keep_frames)
