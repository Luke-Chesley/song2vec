import torch
import matplotlib.pyplot as plt

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)


def normalize_tensor(tensor):
    mean = tensor.mean()

    centered_tensor = tensor - mean

    std = tensor.std()

    normalized_tensor = centered_tensor / std

    return normalized_tensor


def calculate_receptive_field(conv_feature_layers):
    receptive_field = 0
    stride_product = 1

    for layer in conv_feature_layers:
        _, kernel_width, stride = layer
        receptive_field += (kernel_width - 1) * stride_product
        stride_product *= stride

    receptive_field += 1

    return receptive_field


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


from fairseq.utils import buffered_arange, index_put, is_xla_tensor

def sample_negatives(self, y, num, padding_count=None):

    if self.n_negatives == 0 and self.cross_sample_negatives == 0:
        return y.new(0)

    bsz, tsz, fsz = y.shape
    y = y.view(-1, fsz)  # BTC => (BxT)C

    # FIXME: what happens if padding_count is specified?
    cross_high = tsz * bsz
    high = tsz - (padding_count or 0)
    with torch.no_grad():
        assert high > 1, f"{bsz,tsz,fsz}"

        if self.n_negatives > 0:
            tszs = (
                buffered_arange(num)
                .unsqueeze(-1)
                .expand(-1, self.n_negatives)
                .flatten()
            )

            neg_idxs = torch.randint(
                low=0, high=high - 1, size=(bsz, self.n_negatives * num)
            )
            neg_idxs[neg_idxs >= tszs] += 1

        if self.cross_sample_negatives > 0:
            tszs = (
                buffered_arange(num)
                .unsqueeze(-1)
                .expand(-1, self.cross_sample_negatives)
                .flatten()
            )

            cross_neg_idxs = torch.randint(
                low=0,
                high=cross_high - 1,
                size=(bsz, self.cross_sample_negatives * num),
            )
            cross_neg_idxs[cross_neg_idxs >= tszs] += 1

    if self.n_negatives > 0:
        neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
    else:
        neg_idxs = cross_neg_idxs

    if self.cross_sample_negatives > 0 and self.n_negatives > 0:
        neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

    negs = y[neg_idxs.view(-1)]
    negs = negs.view(
        bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
    ).permute(
        2, 0, 1, 3
    )  # to NxBxTxC
    return negs, neg_idxs
