
class StandardScaler:
    """
    Centers Data Around zero
    with Standard Derivation of 1
    more robust with outliers
    """
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


class NormalScaler:
    """
    Rescale values so that each feature's minimum
    value is 0 and their maximum value is 1
    """
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0) - self.min

    def transform(self, x):
        return (x - self.min) / self.max

    def inverse_transform(self, x_scaled):
        return x_scaled * self.max + self.min


def center_data(dataframe, x_names, y_names, anchor_name_x, anchor_name_y):
    """
    Center data in dataframe around an anchor (feature)
    -----------
    dataframe: data as dataframe to be centered
    x_names: list of features - only x coordinates
    y_names: list of features - only y coordinates
    anchor_name_x: name of feature to be centered around - only x coordinates
    anchor_name_y: name of feature to be centered around - only y coordinates
    returns: centered data around anchor
    """

    out = dataframe[x_names].sub(dataframe[anchor_name_x], axis=0)
    out[y_names] = dataframe[y_names].sub(dataframe[anchor_name_y], axis=0)

    return out
