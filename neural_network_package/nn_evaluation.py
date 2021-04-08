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
    #feature_list = np.arange(len(y[0]))
    h = h.argmax(axis=1)
    y = y.argmax(axis=1)
    true_positives, false_positives, true_negatives, false_negatives = determine_true_false_positive_negative(h, y )

    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def plot_confusion_matrix(h, y):
    #feature_list = np.arange(len(y[0]))
    h = h.argmax(axis=1)
    y = y.argmax(axis=1)

    true_positives, false_positives, true_negatives, false_negatives = determine_true_false_positive_negative(h, y)

    confusion_matrix = np.array([[true_negatives, false_negatives],
                                 [false_positives, true_positives]]).astype(int)

    confusion_matrix = confusion_matrix / len(h) * 100
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


def determine_true_false_positive_negative(h, y):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # false positive = event fired, but idle or not right event
    # false negative = no event fired (idle), but there should be an event
    # true positive = correct event fired and not idle
    # true negative = identified idle correctly
    #features=[0,1,2,3]
    #for feature in features:
    #   true_positives += (h == feature) & (y == feature)
    #   false_positives += (h == feature) & (y != feature)
    #   true_negatives += (h != feature) & (y != feature)
    #   false_negatives += (h != feature) & (y == feature)

    num_samples = len(y)
    for sample_idx in range(num_samples):
        current_y_value = y[sample_idx]
        current_h_value = h[sample_idx]

        event_fired = current_h_value != 0

        if event_fired:
            #print("h == event")
            if current_y_value == current_h_value:
                true_positives += 1
            else:
                false_positives += 1
        else:
            #print("h == idle")
            if current_y_value == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    return true_positives, false_positives, true_negatives, false_negatives


h = np.array([[1, 0, 0, 0], #false negative
              [0, 1, 0, 0], #false positive
              [0, 0, 1, 0], #true positive
              [1, 0, 0, 0]])#true negative
y = np.array([[0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 0]])
f1 = f1_score(h, y)
acc = accuracy(h, y)
plot_confusion_matrix(h,y)
