import os
import torch
import pickle
import numpy as np


def read_tep_data(base_path,
                  modes=None,
                  normalization="standardization",
                  return_domain=False,
                  as_list=False,
                  channels_first=True,
                  one_hot_labels=False):
    """Prepares simulations of the Tennessee Eastman Process
    for the training of a deep neural network.

    This function essentially reads data from various modes of
    operation, then normalizes the simulation variables, either
    by a max-min scaling, or by standardization. Other than that,
    this function formats data into a 3D tensor of shape (N, d, T)
    for channels_first=True, and (N, T, d), for channels_first=False.

    Parameters
    ----------
    base_path : str
        Path where the benchmark is rooted (i.e., where the .pickle) files
        are located.
    modes : list, optional
        List of modes of operation to read. Can be any list of integers
        {1, ..., 6}, without repetition.
    normalization : {"standardization", "scaling"}, optional
        If "standardization", standardizes data using the mean and standard
        deviation per simulation variable. If "scaling", scales the data
        using the max and the min of each simulation variable. If any other
        value, it keeps the data as is.
    return_domain : bool, optional
        If True, returns an additional list or vector of domain labels
    as_list : bool, optional
        If True, and if len(modes) > 1, returns the data of each domain
        separately through a list.
    channels_first : bool, optional
        If True, returns data of each domain as (N, d, T). If False,
        return data as (N, T, d).
    one_hot_labels : bool, optional
        If True, returns labels for each data-point as one-hot encoded
        vectors of shape (n_c,).
    """
    if modes is None:
        modes = [1,]

    features, labels, domains = [], [], []
    for mode in modes:
        fname = "TEPDataset_Mode{}.pickle".format(mode)
        with open(os.path.join(base_path, fname), 'rb') as handle:
            data = pickle.load(handle)

        # Get arrays
        X, y = data['Signals'], data['Labels']
        _, n_features = X.shape[1:]

        # Channel-wise normalization
        min_vals, max_vals = X.min(axis=(0, 1)), X.max(axis=(0, 1))
        mean_vals, std_vals = X.mean(axis=(0, 1)), X.std(axis=(0, 1))

        if normalization == 'standardization':
            bias, scale = mean_vals, std_vals
        elif normalization == 'scaling':
            bias, scale = min_vals, (max_vals - min_vals)
        else:
            bias, scale = np.zeros([n_features]), np.ones([n_features])

        for f in range(n_features):
            if np.isclose(scale[f], 0):
                X[..., f] = X[..., f] / bias[f]
            else:
                X[..., f] = (X[..., f] - bias[f]) / scale[f]

        if channels_first:
            X = np.transpose(X, [0, 2, 1])

        features.append(torch.from_numpy(X).float())
        labels.append(torch.from_numpy(y).long())
        domains.append(
            torch.from_numpy(
                np.array([mode] * len(data['Signals']))).long()
        )
        del data

    n_classes = 29

    if as_list:
        if one_hot_labels:
            labels = [
                torch.nn.functional.one_hot(
                    yi, num_classes=n_classes).float()
                for yi in labels]

        if return_domain:
            return features, labels, domains
        return features, labels
    else:
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        domains = torch.cat(domains, dim=0)
        if one_hot_labels:
            labels = torch.nn.functional.one_hot(
                labels, num_classes=n_classes)
        if return_domain:
            return features, labels, domains
        return features, labels
