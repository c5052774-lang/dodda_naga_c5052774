import tensorflow as tf
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import yaml
import os

from .model import BertSequenceClassifier
from .utils import plot_losses, print_class_distribution


def prepare_dataset(tokenizer, dataset, max_length):

    texts = []
    labels = []

    import re

    def _extract_segments(example):
        # Prefer a list-like `story_text` if present
        if "story_text" in example:
            return example["story_text"]

        # Dataset uses `story` (string with grounding tags). Try to extract
        # per-frame segments marked with <gdi>..</gdi>. Fallback to sentence split.
        if "story" in example and isinstance(example["story"], str):
            story = example["story"]
            segments = re.findall(r"<gdi[^>]*>(.*?)</gdi>", story, flags=re.DOTALL)
            if segments:
                return [s.strip() for s in segments if s.strip()]
            # fallback: split into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', story) if s.strip()]
            return sentences

        # Try other common keys
        for key in ("story_texts", "story_sentences", "text"):
            if key in example:
                return example[key]

        return []

    for example in dataset:
        segments = _extract_segments(example)
        if not segments:
            continue
        for idx, sentence in enumerate(segments[:5]):
            texts.append(sentence)
            labels.append(idx)

    print_class_distribution(labels)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2,
        stratify=labels, random_state=42
    )

    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    val_encodings = tokenizer(
        val_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

    return train_encodings, val_encodings, \
           np.array(train_labels), np.array(val_labels)


def train_model(config, experiment_name):

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = load_dataset("daniel3303/StoryReasoning", split="train")

    train_enc, val_enc, train_labels, val_labels = prepare_dataset(
        tokenizer, dataset, config["max_length"]
    )

    model = BertSequenceClassifier(
        config["model_name"],
        num_labels=config["num_labels"],
        dropout_rate=config["dropout_rate"],
        freeze_bert=config["freeze_bert"]
    )

    # TF sometimes rejects numpy types or YAML-parsed values; ensure float
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=float(config.get("learning_rate", 0.0))
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x=(train_enc["input_ids"], train_enc["attention_mask"]),
        y=train_labels,
        validation_data=(
            (val_enc["input_ids"], val_enc["attention_mask"]),
            val_labels
        ),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=1
    )

    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]

    os.makedirs("results/figures", exist_ok=True)
    plot_losses(
        history.history["loss"],
        history.history["val_loss"],
        f"results/figures/{experiment_name}_loss.png"
    )

    print(f"\nFinal Train Loss: {train_loss:.4f}")
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Final Val Accuracy: {val_acc:.4f}")

    return train_loss, val_loss, val_acc