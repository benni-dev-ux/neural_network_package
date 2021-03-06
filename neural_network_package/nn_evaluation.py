import numpy as np


def accuracy(h, y):
    """calculates accuracy for binary classification"""
    return np.mean(np.round(h) == y)


def accuracy_multiclass(h, y):
    """calculates accuracy for multiclass classification"""
    return (h.argmax(axis=1) == y.argmax(axis=1)).mean()


def f1_score(h, y, true_negative_value=0):
    """calculates the accuracy from the precision and recall of the test"""
    h = h.argmax(axis=1)
    y = y.argmax(axis=1)
    true_positives, false_positives, true_negatives, false_negatives = determine_true_false_positive_negative(h, y,
                                                                                                              true_negative_value)

    if true_positives == 0:
        print("no true positives")
        return 0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def determine_true_false_positive_negative(h, y, true_negative_value):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    num_samples = len(y)
    for sample_idx in range(num_samples):
        current_y_value = y[sample_idx]
        current_h_value = h[sample_idx]

        event_fired = current_h_value != true_negative_value

        if event_fired:
            if current_y_value == current_h_value:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if current_y_value == true_negative_value:
                true_negatives += 1
            else:
                false_negatives += 1

    return true_positives, false_positives, true_negatives, false_negatives
