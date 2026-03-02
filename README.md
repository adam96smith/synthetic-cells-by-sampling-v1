# Synthetic Cells by Sampling: Full Pipeline for Cell Segmentation

This repository provides a complete pipeline for generating **synthetic training images of A549 cells with branching filopodia**, training a **segmentation model**, and evaluating the trained model on **real data**.
The goal is to leverage synthetic data to improve segmentation performance on real-world images.

The pipeline is organized into **four main execution stages**, each controlled by a Bash script and parameterized via configuration files.

---

## Repository Structure

```
.
├── Fluo-C3DH-A549/               # Test data from Cell Tracking Challenge
├── Fluo-C3DH-A549-train/         # Annotated training data from Cell Tracking Challenge
├── main/                         # All executable scripts (run commands from here)
│   ├── scripts/
│   │   └── default/
│   │       ├── Sampling.sh       # Sample intensities from annotated data (eg. Fluo-C3DH-A549-train/)
│   │       ├── ModelTrain.sh     # Train segmentation model on dataset stored in synthetic_data/
│   │       ├── RunFinal.sh       # Run inference on real data (Fluo-C3DH-A549/ or Fluo-C3DH-A549-train/)
│   │       └── ...
│   │   └── A549/
│   │       ├── TrainData.sh      # Generate dataset with synthetic images with labels
│   │       └── ...
│   ├── config/                   # Configuration files for each stage
│   ├── data_generator/           # Scripts for sampling, and shape + image generation
│   ├── model_codes/              # Scripts for model training and inference
│   ├── synthetic_data/           # Where synthetic training images are stored
│   ├── models/                   # Where trained models are stored
│   ├── utils/
│   └── ...
```

> **Important:**
> All scripts must be executed **from the `main/` directory**, for example:
>
> ```bash
> cd main
> bash scripts/default/Sampling.sh
> ```

---

## Pipeline Overview

The full workflow consists of the following steps:

1. **Synthetic Data Sampling**
2. **Training Data Generation**
3. **Model Training**
4. **Final Evaluation / Inference**

Each step:

* Is executed via a Bash script
* Loads parameters from configuration files located in `main/config/`
* Produces outputs that feed into the next stage

---

## 1. Synthetic Data Sampling

**Script:**

```bash
bash scripts/default/Sampling.sh
```

### Purpose

Generates synthetic imaging data and corresponding segmentation labels.
This step controls the **diversity**, **complexity**, and **distribution** of the synthetic dataset.

### Key Outputs

* Synthetic images
* Ground-truth segmentation masks
* Metadata describing generation parameters

### Configuration Files

Located in:

```
main/config/
```

#### Parameter Effects (Examples)

| Parameter      | Description                          | Effect on Output               |
| -------------- | ------------------------------------ | ------------------------------ |
| `num_samples`  | Number of synthetic images generated | Controls dataset size          |
| `noise_level`  | Amount of simulated noise            | Affects realism and robustness |
| `object_count` | Number of objects per image          | Controls scene complexity      |
| `random_seed`  | Seed for random sampling             | Ensures reproducibility        |

> Use this section to briefly document **how each parameter influences the generated images**.

---

## 2. Training Data Preparation

**Script:**

```bash
bash scripts/default/TrainData.sh
```

### Purpose

Transforms raw synthetic data into a format suitable for model training.
This may include resizing, normalization, augmentation, and dataset splitting.

### Key Outputs

* Processed images
* Processed segmentation masks
* Train / validation splits

### Configuration Files

Located in:

```
main/config/
```

#### Parameter Effects (Examples)

| Parameter      | Description                  | Effect on Output               |
| -------------- | ---------------------------- | ------------------------------ |
| `image_size`   | Target resolution            | Affects model input dimensions |
| `augmentation` | Enable/disable augmentations | Improves generalization        |
| `train_split`  | Training set fraction        | Controls validation size       |

---

## 3. Model Training

**Script:**

```bash
bash scripts/default/ModelTrain.sh
```

### Purpose

Trains the segmentation model using the prepared synthetic dataset.

### Key Outputs

* Trained model checkpoints
* Training logs
* Loss and metric curves

### Configuration Files

Located in:

```
main/config/
```

#### Parameter Effects (Examples)

| Parameter       | Description           | Effect on Output                 |
| --------------- | --------------------- | -------------------------------- |
| `learning_rate` | Optimizer step size   | Affects convergence stability    |
| `batch_size`    | Samples per iteration | Impacts memory usage and speed   |
| `num_epochs`    | Training duration     | Controls final model performance |
| `loss_function` | Loss definition       | Influences segmentation behavior |

---

## 4. Final Evaluation / Inference

**Script:**

```bash
bash scripts/default/RunFinal.sh
```

### Purpose

Runs the trained model on **real data** or held-out datasets to evaluate performance and generate final predictions.

### Key Outputs

* Predicted segmentation masks
* Evaluation metrics
* Visualization outputs (optional)

### Configuration Files

Located in:

```
main/config/
```

#### Parameter Effects (Examples)

| Parameter         | Description              | Effect on Output          |
| ----------------- | ------------------------ | ------------------------- |
| `checkpoint_path` | Model checkpoint to load | Selects trained model     |
| `threshold`       | Segmentation threshold   | Controls mask sensitivity |
| `output_format`   | Save format              | Affects downstream usage  |

---

## Execution Order Summary

```bash
cd main

bash scripts/default/Sampling.sh
bash scripts/default/TrainData.sh
bash scripts/default/ModelTrain.sh
bash scripts/default/RunFinal.sh
```


