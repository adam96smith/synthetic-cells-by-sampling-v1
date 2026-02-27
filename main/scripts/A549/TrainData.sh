#!/bin/bash

DATASET_NAME='A549'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# # Training Data
# # Get Sampler for all labelled data
# python data_generator/custom_$DATASET_NAME/GeneratorMask.py \
#     --save-path synthetic_data/$DATASET_NAME/ \
#     --N 5 \

# # Create Images by Partition Sampling
# python data_generator/default/GeneratorImage.py \
#     --dataset-id $DATASET_NAME \
#     --mask-dir synthetic_data/$DATASET_NAME/ \
#     --sampler-dir data_$DATASET_NAME/ \
#     --sub-folder custom_texture/ \
#     --config config/$DATASET_NAME/synth_parameters_A.yaml

# python data_generator/default/GeneratorImage.py \
#     --dataset-id $DATASET_NAME \
#     --mask-dir synthetic_data/$DATASET_NAME/ \
#     --sampler-dir data_$DATASET_NAME/ \
#     --sub-folder aug_texture/ \
#     --config config/$DATASET_NAME/synth_parameters_B.yaml 

# python data_generator/default/GeneratorImage.py \
#     --dataset-id $DATASET_NAME \
#     --mask-dir synthetic_data/$DATASET_NAME/ \
#     --sampler-dir data_A549_slice/ \
#     --sub-folder aug_texture_slice/ \
#     --config config/$DATASET_NAME/synth_parameters_B.yaml 

python data_generator/default/GeneratorImage.py \
    --dataset-id $DATASET_NAME \
    --mask-dir synthetic_data/$DATASET_NAME/ \
    --sampler-dir data_$DATASET_NAME/ \
    --sub-folder blur_texture/ \
    --config config/$DATASET_NAME/synth_parameters_C.yaml 

python data_generator/default/GeneratorImage.py \
    --dataset-id $DATASET_NAME \
    --mask-dir synthetic_data/$DATASET_NAME/ \
    --sampler-dir data_A549_slice/ \
    --sub-folder blur_texture_slice/ \
    --config config/$DATASET_NAME/synth_parameters_C.yaml 

# python data_generator/custom_$DATASET_NAME/GeneratorCurveSeg.py \
#     --input-dir synthetic_data/$DATASET_NAME/ \
#     --dataset-id $DATASET_NAME 