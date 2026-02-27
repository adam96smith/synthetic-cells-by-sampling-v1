#!/bin/bash

DATASET_NAME='DICTY'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/

# # Train Model

python model_codes/default/ModelTrain.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --train-dir synthetic_data/$DATASET_NAME/custom_texture/ \
    --eval-dir Fluo-$DATASET_NAME-train/ \
    --model-dir models/$DATASET_NAME/ \
    --model-name test_model \
    --config config/$DATASET_NAME/model_train.yaml