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
