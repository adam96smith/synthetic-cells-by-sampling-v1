#!/bin/bash

DATASET_NAME='A549'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# Real Labelled Data
python data_generator/default/FormatRealData.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --data-dir Fluo-C3DH-A549-train/

# # Create Syntetic Images from Ground Truth Labels
# python data_generator/default/GeneratorImage.py \
#     --dataset-id $DATASET_NAME \
#     --mask-dir synthetic_data/$DATASET_NAME_real/ \
#     --sampler-dir data_generator/data_$DATASET_NAME/ \
#     --resample \
#     --sub-folder custom_texture/ \
#     --config config/$DATASET_NAME/synth_parameters.yaml


# python data_generator/custom_A549/GeneratorCurveSeg.py \
#     --input-dir synthetic_data/$DATASET_NAME_real/ \
#     --dataset-id $DATASET_NAME 