def normalize(values):
    """
    Rescale values so that each feature's minimum
    value is 0 and their maximum value is 1
    """

    values = values - values.min(axis=0)
    max_values = values.max(axis=0)
    values = values / max_values

    return values


def standardize(values):
    """
    Centers Data Around zero
    with Standard Derivation of 1
    """

    values = values - values.mean(axis=0)
    if values.std(axis=0) != 0:
        values = values / values.std(axis=0)

    return values


class StandardScaler:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

        # the standard deviation can be 0, which provokes
        # devision-by-zero errors; let's omit that:
        self.std[self.std == 0] = 0.00001

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x_scaled):
        return x_scaled * self.std + self.mean
