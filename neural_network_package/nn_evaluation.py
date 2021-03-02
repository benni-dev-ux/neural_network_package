import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_history_to_image(history, title, filename):
    """
    plots error history to an image file
    filename: name for saved image
    """
    fig, ax = plt.subplots()
    ax.plot(history, label=title)
    # "zoom in" on 90% of the values:
    # upper_ylim = np.quantile(error_history, 0.99)
    # ax.set_ylim(-0.001, upper_ylim)

    ax.set_xlabel("Iterations")
    ax.set_ylabel(title)
    ax.set_title(title)
    fig.legend()
    fig.savefig(filename)


def accuracy(h, y):
    return np.mean(np.round(h) == y)


def accuracy_multiclass(h, y):
    return (h.argmax(axis=1) == y.argmax(axis=1)).mean()


def f1_score(h, y):
    h = h.argmax(axis=1)
    h = np.identity(10, dtype=int)[h]
    true_positives = (h == 1) & (y == 1)
    false_positives = (h == 1) & (y == 0)
    false_negatives = (h == 0) & (y == 1)

    precision = np.sum(true_positives) / \
        (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / \
        (np.sum(true_positives) + np.sum(false_negatives))

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def plot_confusion_matrix(h, y):
    h = h.argmax(axis=1)
    h = np.identity(10, dtype=int)[h]

    true_positives = (h == 1) & (y == 1)
    false_positives = (h == 1) & (y == 0)
    true_negatives = (h == 0) & (y == 0)
    false_negatives = (h == 0) & (y == 1)

    confusion_matrix = np.array([[true_negatives.sum(), false_negatives.sum()],
                                 [false_positives.sum(), true_positives.sum()]]).astype(int)

    # visualize the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = ["true negatives\n %d", "false negatives\n %d",
              "false positives\n %d", "true positives\n %d"]
    label_values = np.array([l % v for l, v in zip(
        labels, confusion_matrix.flat)]).reshape(2, 2)

    sns.heatmap(confusion_matrix, annot=label_values, fmt="", ax=ax)
    ax.set_xlabel("ground truth")
    ax.set_ylabel("predicted")
    ax.set_title("confusion matrix")

    # fix for display bug
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
