# mouse-face-detector
Official repository containing our mouse face detector used in the paper **"Automatic pain face analysis in mice: Applied to a varied dataset with non-standardized conditions"** (not yet published).

## Overview
This repository provides a pipeline to transform raw mouse behavior videos into high-quality face image crops ready for **Mouse Grimace Scale (MGS)** rating.

## Processing Pipeline
The script `extract_MGS_frames.py` processes videos through the following stages:

1. **Facial Feature Detection**: Uses the **DeepLabCut (DLC) API** to identify and track key facial landmarks.
2. **Heuristic Frame Extraction**: The script selects "rateable" frames based on:
    * **Feature Visibility**: Priority is given to frames where the maximum number of facial features are detected with high confidence.
    * **Blur Filtering**: Implementation of a variance of Laplacian method as well as Fast Fourier Transform to ensure image sharpness.
    * **Temporal Diversity**: Ensures selected frames are not too close in time to avoid redundant data points.

---

## Getting Started

### Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/mouse-face-detector.git](https://github.com/your-username/mouse-face-detector.git)
cd mouse-face-detector

# Install dependencies
pip install -r requirements.txt
```

### Other requirements
Requires a trained DeepLabCut model tracking these five mouse facial features ('_l' and '_r' for left and right from the mouse's point of view):
* 'nose'
* 'eye_l'
* 'ear_l'
* 'eye_r'
* 'ear_r'

For information on DeepLabCut see <https://www.mackenziemathislab.org/deeplabcut>.

### Example call

```bash
python extract_MGS_frames.py /home/username/example.mp4 -m /home/username/dlc_models/mouse-facial-features/config.yaml
```
