#!/bin/bash

DATASET_NAME='A549'
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$(realpath "$SCRIPT_DIR/../../..")"/
VERSION='model_best.pt'
TEST_DIR='Fluo-C3DH-A549/'

# Run Final for CTC
## Volume Models
python model_codes/default/ModelRun.py \
    --data-root $DATASET_PATH \
    --dataset-id $DATASET_NAME \
    --test-dir $TEST_DIR \
    --model-dir models/$DATASET_NAME/ \
    --model-name submitted_model \
    --model-version $VERSION \
    --final \
    --config config/$DATASET_NAME/model_train.yaml 