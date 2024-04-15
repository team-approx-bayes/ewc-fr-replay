# Required environmental variables for the script:
export IMAGENET_DIR=./src/data/imagenet
export WRITE_DIR=./src/data/imagenet_ffcv_400_1.0_90
export FFCV_IMAGENET_ROOT=../ffcv-imagenet

# Starting in the root of the Git repo:
cd $FFCV_IMAGENET_ROOT;

# Serialize images with:
# - 400px side length maximum
# - 100% JPEG encoded
# - quality=90 JPEGs
./write_imagenet.sh 400 1.0 90