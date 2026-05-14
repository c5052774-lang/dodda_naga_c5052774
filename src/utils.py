import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter


def compute_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1)
    return accuracy_score(labels, preds)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def print_class_distribution(labels):
    counter = Counter(labels)
    total = len(labels)
    print("\nClass Distribution:")
    for k in sorted(counter.keys()):
        print(f"Class {k}: {counter[k]} ({counter[k]/total:.2%})")