# PIVNO stereo baseline

This branch adapts PIVNO to supervised rectified stereo matching:

- full-resolution grayscale left/right images are passed to the model;
- the encoder learns 1/4-resolution features;
- the coordinate-query SR module is removed;
- a one-channel disparity head predicts at 1/4 resolution;
- bilinear interpolation and horizontal scale correction restore the input size.

## Dataset list

Each line contains three tab-separated paths: left image, right image, and disparity.

Disparity can be NPY, PFM, PNG/TIFF, or legacy FLO (horizontal component
only). Set disparity_scale to 256.0 for KITTI-style uint16 PNG values, or
1.0 when files already contain disparity in pixels. Invalid ground-truth
disparity should be non-positive or non-finite.

Update the list paths in configs/train_stereo.yaml and configs/test_stereo.yaml.
Then run train.py, test.py, or demo.py with the corresponding config/model.

Demo image names must contain left/right or img1/img2. It saves raw float
disparity as NPY and a color visualization as PNG.

Old PIV checkpoints are not shape-compatible with the one-channel disparity
head; train a new stereo checkpoint.
