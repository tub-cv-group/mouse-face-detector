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
    
    This applies a trained DLC model to a new video.
"""

import deeplabcut as dlc
import argparse
import os

def find_pickle(vid):
    picklefile = None
    for f in os.listdir(os.path.dirname(vid)):
        if '_bx.pickle' in f and os.path.basename(vid).rsplit('.',1)[0] in f:
            if picklefile is not None:
                print("Error: Found several matching .pickle files in {}. Exiting.".format(os.path.dirname(vid)))
                exit(1)
            picklefile = os.path.join(os.path.dirname(vid), f)
    return picklefile

def find_detections(vid):
    det = None
    for f in os.listdir(os.path.dirname(vid)):
        if '.h5' in f and os.path.basename(vid).rsplit('.',1)[0] in f:
            if det is not None:
                print("Error: Found several matching .h5 files in {}. Exiting.".format(os.path.dirname(vid)))
                exit(1)
            det = os.path.join(os.path.dirname(vid), f)
    return det

"""
    returns path to resulting file. If nothing was detected, because there were no mice in the video or the video
    file was corrupted, the resulting file does not exist.
"""
def main(vid, config_path):
    dlc.analyze_videos(config_path, [vid], save_as_csv=True)
    result_file = find_detections(vid)
    return result_file

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Applies an already trained DLC model to a video. Resulting .h5 file will be put next to the video.")
    argparser.add_argument("video", help="Video file on which to apply the model.")
    argparser.add_argument("config", help="Path to the config.yaml file of the trained DLC model.")
    a = argparser.parse_args()
    res = main(a.video, a.config)
    print("Result:", res)
