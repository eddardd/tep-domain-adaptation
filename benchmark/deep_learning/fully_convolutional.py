import torch


class FullyConvolutionalEncoder(torch.nn.Module):
    r"""Tennessee Eastman Fully Convolutional Encoder.

    Consists of a FCN for extracting features
    from TEP raw signals.

    Parameters
    ----------
    n_features : int, optional (default=51)
        Number of features in the raw data. Corresponds
        to the number of sensors in the time series.
    n_time_steps : int, optional (default=600)
        Number of time steps in each time series.
    batch_norm : bool, optional (defualt=False)
        If True, applies batch normalization on conv blocks.
    instance_norm : bool, optional (defualt=True)
        If True, applies instance normalization on conv blocks.
    """
    def __init__(self,
                 n_features=51,
                 n_time_steps=600,
                 batch_norm=False,
                 instance_norm=True):
        super(FullyConvolutionalEncoder, self).__init__()

        x = torch.randn(16, n_features, n_time_steps)

        if batch_norm:
            self.main = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=128,
                                kernel_size=9,
                                padding='same'),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same'),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                padding='same'),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(inplace=True),
            )
        elif instance_norm:
            self.main = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=128,
                                kernel_size=9,
                                padding='same'),
                torch.nn.InstanceNorm1d(num_features=128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same'),
                torch.nn.InstanceNorm1d(num_features=256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                padding='same'),
                torch.nn.InstanceNorm1d(num_features=128),
                torch.nn.ReLU(inplace=True),
            )
        else:
            self.main = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=n_features,
                                out_channels=128,
                                kernel_size=9,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=128,
                                out_channels=256,
                                kernel_size=5,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv1d(in_channels=256,
                                out_channels=128,
                                kernel_size=3,
                                padding='same'),
                torch.nn.ReLU(inplace=True),
            )
        with torch.no_grad():
            h = self.main(x)
            h = h.mean(dim=-1)
        self.n_out_feats = h.shape[-1]

    def forward(self, x):
        return self.main(x).mean(dim=-1)
