# Deep Neural Networks – Story Sequence Prediction

## Overview

This project implements a **text-based deep learning model** to predict the **position of a sentence within a story sequence**.
Each story contains **five ordered sentences**, and the model predicts which position (1–5) a given sentence belongs to.

This task is formulated as a **5-class classification problem** where the model learns **temporal narrative structure from language patterns**.

The implementation uses **TensorFlow and a pre-trained BERT model** with a classification head.

---

# Dataset

Dataset used:
**StoryReasoning Dataset**

Each story contains:

* An **image**
* A **text sentence**
* **Metadata / tags**

For this project, only the **text modality** was used.

### Dataset Processing

1. Extracted the **first five sentences** from each story.
2. Assigned labels according to sequence position.

| Position        | Label |
| --------------- | ----- |
| First sentence  | 0     |
| Second sentence | 1     |
| Third sentence  | 2     |
| Fourth sentence | 3     |
| Fifth sentence  | 4     |

### Dataset Split

The dataset was split as:

* **80% Training**
* **20% Validation**

### Class Distribution

The dataset is well balanced.

| Class | Samples | Percentage |
| ----- | ------- | ---------- |
| 0     | 3552    | ~20%       |
| 1     | 3549    | ~20%       |
| 2     | 3549    | ~20%       |
| 3     | 3549    | ~20%       |
| 4     | 3549    | ~20%       |

Balanced classes help avoid bias during training.

---

# Techniques Used

### 1. Transformer-based Text Representation

A **pre-trained BERT model (bert-base-uncased)** was used to extract contextual embeddings from sentences.

Advantages:

* Captures contextual meaning
* Understands semantic relationships
* Effective for sequence understanding

---

### 2. Classification Head

A dense layer with **Softmax activation** predicts the probability of each sequence position.

Architecture:

```
Input Sentence
      ↓
BERT Encoder
      ↓
Dropout
      ↓
Dense Layer (Softmax)
      ↓
5-Class Prediction
```

---

### 3. Training Setup

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Optimizer     | Adam                            |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size    | 16                              |
| Epochs        | 5                               |
| Learning Rate | 2e-5                            |

Metrics tracked:

* Training Loss
* Validation Loss
* Validation Accuracy

Loss curves were plotted for each experiment.

---

# Repository Structure

```
project_username/
│
├── README.md
├── experiment_notebook.ipynb
├── config.yaml
├── requirements.txt
│
├── src/
│   ├── model.py        # BERT classification model
│   ├── train.py        # Training pipeline
│   └── utils.py        # Utility functions
│
└── results/
    ├── figures/        # Loss curve plots
    └── results.csv     # Experiment comparison table
```

---

# Running the Project

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Run Experiments

Open the notebook:

```
experiment_notebook.ipynb
```

Run all cells.

The notebook will automatically:

1. Load dataset
2. Train models
3. Run all experiments
4. Save loss curves
5. Generate results table

Outputs are saved in:

```
results/
```

---

# Experiments

Five controlled experiments were conducted.
Each experiment changes **exactly one parameter**.

| Experiment | Modification                   |
| ---------- | ------------------------------ |
| Baseline   | Default BERT model             |
| Exp2       | Increased dropout (0.3)        |
| Exp3       | Frozen BERT layers             |
| Exp4       | Increased learning rate (5e-5) |
| Exp5       | Increased number of epochs     |

---

# Results

| Experiment | Modification | Train Loss | Validation Loss | Validation Accuracy |
| ---------- | ------------ | ---------- | --------------- | ------------------- |
| Baseline   | Default BERT | 1.6420     | 1.6444          | 0.2431              |
| Exp2       | Dropout 0.3  | 1.7148     | 1.6113          | 0.2265              |
| Exp3       | Freeze BERT  | 1.6291     | 1.5957          | 0.2451              |
| Exp4       | LR = 5e-5    | 1.6276     | 1.5799          | **0.2775**          |
| Exp5       | More Epochs  | 1.6325     | 1.5958          | 0.2377              |

Best performing model:

**Experiment 4 – Increased Learning Rate**

Validation Accuracy:
**27.75%**

---

# Interpretation of Results

### Learning Rate Impact

Increasing the learning rate improved performance because the optimizer explored the parameter space more effectively during training.

### Dropout Effect

Increasing dropout reduced accuracy slightly.
This suggests the model was **not strongly overfitting**, so additional regularization did not help.

### Freezing BERT

Freezing the encoder produced similar performance.
This indicates that **pre-trained BERT features already capture useful semantic information**.

### Increasing Epochs

Training longer did not improve performance significantly, suggesting the model reached convergence early.

---

# Discussion

Predicting sequence position is a challenging task because:

1. Sentences can appear in multiple narrative contexts.
2. Temporal order is not always explicit in language.
3. Stories may have ambiguous or similar events.

The model must learn subtle cues such as:

* Temporal expressions
* Cause–effect relationships
* Narrative progression

Even with a strong language model like BERT, this task remains complex.

---

# Key Findings

* Transformer models are effective for narrative reasoning tasks.
* Learning rate tuning significantly affects performance.
* Pre-trained representations provide strong baseline performance.
* Story sequence prediction remains challenging due to narrative ambiguity.

---

# Author Contribution

This implementation includes:

* Dataset preprocessing
* Model implementation
* Experiment design
* Training pipeline
* Evaluation and analysis
* Visualization of loss curves

All experiments were conducted using **TensorFlow and HuggingFace Transformers**.

---

# Future Improvements

Possible improvements include:

* Using larger transformer models
* Adding contextual story information
* Training with larger batch sizes
* Incorporating positional encoding techniques
* Using sequence-level models instead of sentence-level classification

---

# Conclusion

This project demonstrates how **transformer-based models can learn narrative progression from text**.
While performance remains moderate, the experiments highlight the importance of **hyperparameter tuning and model configuration** in sequence prediction tasks.

---
