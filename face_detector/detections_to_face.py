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

    Takes detections of mice facial features, that were done using Deeplabcut applied to a video.
    Uses heuristics to find the face bounding boxes.

    Also returns time points (indices of the frames) when the candidate faces happened in the video.

    Chooses candidate detections like this:
        - Takes those detections, where the most facial features were found (ideally all)
        - Out of those: Takes those, where the sum of likelihoods (after thresholding) is the biggest
        - Always ensures, that there is at least a distance of 80 frames between selected frames
        - Returns the requested number of candidates (at most)
"""

import argparse
import os
import pandas as pd
import numpy as np
from .cvpybox import bbox as bb

def get_candidates(num_candidates, num_feats_detected, lh_sum, min_time_diff):
    """
        @return a pd.DataFrame of the features locations and likelihoods as value. The index of the DataFrame is the frame number in the original video (i.e. the time point)
    """
    num_face_features = 5 # (eyes, ears, nose)
    # add from those detections with the highest number of features detected in order of likelihood sum num_candidates to the list
    idx_with_this_num = []
    while len(idx_with_this_num) == 0:
        idx_with_this_num = np.where(num_feats_detected==num_face_features)[0]
        num_face_features -= 1
    idx_with_this_num = idx_with_this_num[np.argsort(lh_sum[idx_with_this_num])[::-1]] # indices sorted descending by likelihood
    cand_idx = np.array(idx_with_this_num)
    # ensure, that candidates are not too close together in time
    sorted_cand_idx = np.sort(cand_idx)
    cand_idx_time = [sorted_cand_idx[0]]
    i = 1
    while i < len(sorted_cand_idx):
        new_one = sorted_cand_idx[i]
        last_one = cand_idx_time[-1]
        if np.abs(new_one-last_one) > min_time_diff: # if the next one coming in is not too close in time to the last one added
            cand_idx_time.append(new_one)
        else: # if the next one coming in IS too close in time to the last one added
            if num_feats_detected[new_one] > num_feats_detected[last_one]: # if the new one has more features
                cand_idx_time[-1] = new_one # replace the last one by the new one
            elif num_feats_detected[last_one] > num_feats_detected[new_one]:
                pass # keep the last one or if both have the same number of features detected:
            elif lh_sum[new_one] > lh_sum[last_one]: # if the likelihood sum of the new one is better than the last one added
                cand_idx_time[-1] = new_one # replace the last one by the new one
        i += 1
    cand_idx = np.array([entry for entry in cand_idx if entry in cand_idx_time])
    to_take = min(num_candidates, len(cand_idx)-1)
    cand_idx = cand_idx[np.argpartition(lh_sum[cand_idx], to_take)[-to_take:]]
    # if not enough are there yet, do the same for the following numbers of detections
    if len(cand_idx) < num_candidates:
        cand_idx_sorted_by_time = np.sort(cand_idx)
        while num_face_features >= 0:
            if len(cand_idx_sorted_by_time) >= num_candidates: break
            idx_with_this_num = np.where(num_feats_detected==num_face_features)[0]
            idx_with_this_num = idx_with_this_num[np.argsort(lh_sum[idx_with_this_num])[::-1]] # indices sorted descending by likelihood
            for new_one in idx_with_this_num:
                pos = np.searchsorted(cand_idx_sorted_by_time, new_one)
                if pos > 0:
                    existing_one = cand_idx_sorted_by_time[pos-1]
                else:
                    existing_one = -min_time_diff-1
                if pos < len(cand_idx_sorted_by_time):
                    next_ex_one = cand_idx_sorted_by_time[pos]
                else:
                    next_ex_one = np.inf
                if np.abs(new_one-existing_one) < min_time_diff: # if the new one coming in is too close in time to the one before it in time
                    if num_feats_detected[new_one] > num_feats_detected[existing_one]: # if the new one has more features
                        cand_idx_sorted_by_time[pos] = new_one # replace the last one by the new one
                    elif num_feats_detected[existing_one] > num_feats_detected[new_one]:
                        pass # keep the last one or if both have the same number of features detected:
                    elif lh_sum[new_one] > lh_sum[existing_one]: # if the likelihood sum of the new one is better than the last one added
                        cand_idx_sorted_by_time[pos] = new_one # replace the last one by the new one
                elif np.abs(new_one-next_ex_one) < min_time_diff: # if the new one coming in is too close in time to the one after it in time
                    if num_feats_detected[new_one] > num_feats_detected[next_ex_one]: # if the new one has more features
                        cand_idx_sorted_by_time[pos] = new_one # replace the last one by the new one
                    elif num_feats_detected[next_ex_one] > num_feats_detected[new_one]:
                        pass # keep the last one or if both have the same number of features detected:
                    elif lh_sum[new_one] > lh_sum[next_ex_one]: # if the likelihood sum of the new one is better than the last one added
                        cand_idx_sorted_by_time[pos] = new_one # replace the last one by the new one
                else: # the new one is not too close to any of them
                    cand_idx_sorted_by_time = np.insert(cand_idx_sorted_by_time, pos, new_one)
                if len(cand_idx_sorted_by_time) >= num_candidates: break
            num_face_features -= 1
        cand_idx = cand_idx_sorted_by_time
    return cand_idx

def faces_from_candidate_detections(cands, min_bb_area, lh_thresh, scale_bbox_min, scale_bbox_max, shape):
    required_feats = ['nose', 'eye_l', 'ear_l', 'eye_r', 'ear_r']
    for f in required_feats:
        if not f in cands.columns.levels[0]:
            print(f"Error: The DLC results don't seem to contain the required feature names.\n\tRequired:{required_feats}\n\tFound:{cands.columns.levels[0]}\n\tExiting.")
            exit(1)
    faces = []
    time_points = []
    for idx, row in cands.iterrows():
        nose_detected = row['nose']['likelihood'] > lh_thresh
        an_eye_detected = row['eye_l']['likelihood'] > lh_thresh or row['eye_r']['likelihood'] > lh_thresh
        an_ear_detected = row['ear_l']['likelihood'] > lh_thresh or row['ear_r']['likelihood'] > lh_thresh
        validity = sum([nose_detected, an_eye_detected, an_ear_detected])
        validity = 3 if nose_detected and an_ear_detected else validity
        if validity > 1:
            coords = np.empty([5,2]); coords[:] = np.NaN
            for i,f in enumerate(required_feats):
                if row[f]['likelihood'] > lh_thresh:
                    coords[i][0] = row[f]['y']
                    coords[i][1] = row[f]['x']
            # filter invalid coords
            filtered = coords[~np.any(np.isnan(coords), axis=1)]
            bbox = bb.from_coords(filtered)
            bbox = bb.scale(bbox, scale_bbox_max) if validity < 3 else bb.scale(bbox, scale_bbox_min)
            bbox = bb.make_square(bbox)
            bbox = bb.fit_with_moving(bbox, (shape[1],shape[0])) # cvpybox expects shape as (height, width)
            h, w = bb.hw(bbox)
            area = h*w
            if area >= min_bb_area:
                faces.append(bbox) # bbox is tuple (r_min, r_max, c_min, c_max) with 'r': row and 'c': column
                time_points.append(idx)
            else:
                print(f"Candidate {idx} was not taken, because the face bounding box was too small.")
        else:
            print(f"Candidate {idx} was not taken, because its validity was not > 1.")
    return (time_points, faces)

def main(feat_file, num_candidates=100, lh_thresh=.4, shape=[1936, 1216]):
    if feat_file.endswith('csv'):
        feats = pd.read_csv(feat_file, header=[0,1,2,3])
    else:
        feats = pd.read_hdf(feat_file)
    multiidx = feats.columns.levels
    feats = feats[multiidx[0][0]] # get rid of scorer index, it's the same for the whole data anyways
    if num_candidates >= feats.shape[0]: # if more candidates are requested than there are frames, just take all of the frames
        cands = feats
    elif num_candidates < 1:
        print(f"ERROR: trying to get the \"{num_candidates} best detections\".")
        exit(1)
    else:
        # compute the number of features, that were detected and the sums of the likelihoods for each mouse
        num_feats_detected = np.zeros(feats.shape[0])
        lh_sum = np.zeros(feats.shape[0])
        for index, row in feats.iterrows():
            lh = list(filter(lambda lh: lh and lh>lh_thresh, row.xs('likelihood', level=1, drop_level=False).values))
            num_feats_detected[index] = len(lh)
            lh_sum[index] = sum(lh)

        cand_idx = dict()
        min_time_diff = min(40*2, feats.shape[0]/num_candidates/2) # with video recorded at 40 fps this will result in at least two seconds being between two detected faces
        cands = get_candidates(num_candidates, num_feats_detected, lh_sum, min_time_diff)
        cands = feats.loc[np.sort(cands)]

    min_bb_area = 50
    scale_bbox_min = (1.66, 1.66)
    scale_bbox_max = (2.35, 2.35)
    result = faces_from_candidate_detections(cands, min_bb_area, lh_thresh, scale_bbox_min, scale_bbox_max, shape)
    return result

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Computes Face Detections from Facial Features.")
    argparser.add_argument("features_file", help=".csv or .h5 file with the detections.")
    argparser.add_argument("-n", "--num_candidates", help="The maximum number of face detections to return for each mouse. (Will be less than this, if there were fewer frames) Default: 1000", default=1000, type=np.int)
    argparser.add_argument("-l", "--likelihood_thresh", help="The threshold under which a detection is not used. The default value is chosen by experience. Don't choose this threshold too high, it will make the bounding box worse. The program in any case favors higher likelihoods - independent of the threshold. Default: 0.45", default=.45, type=np.float)
    argparser.add_argument("-s", "--shape", help="List of two values: width and height of the image data. Default: [1936, 1216]", default=[1936, 1216])
    a = argparser.parse_args()
    res = main(a.features_file, a.num_candidates, a.likelihood_thresh, a.shape)
    print(res)
