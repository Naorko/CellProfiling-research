import os
from math import floor, ceil

from matplotlib import pyplot as plt
import numpy as np

from data_layer.channels import Channels

CMAP = 'gray'  # 'viridis'


def show_input_and_target(input, target=None, pred=None, title='', save_dir=None, target_channel=None):
    input_c = input.shape[0]
    target_c = 0 if target is None else target.shape[0]
    pred_c = 0 if pred is None else pred.shape[0]
    num_images = input_c + target_c + pred_c
    images_in_row = 5 if num_images > 6 else 3

    num_rows, num_cols = ceil(num_images / images_in_row), images_in_row

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*2*1.2, num_rows*2*1.3))
    if num_rows == 1 or num_cols == 1:
        ax = np.expand_dims(ax, axis=0)

    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    chan_name_idx = 0

    for c in range(input_c):
        ax[c // num_cols, c % num_cols].imshow(input[c, :, :], cmap=CMAP)
        chan = list(Channels)[chan_name_idx]
        if chan == target_channel and input_c != target_c:
            chan_name_idx += 1
            chan = list(Channels)[chan_name_idx]

        ax[c // num_cols, c % num_cols].set_title(chan.name)
        chan_name_idx += 1

    ax_title = target_channel.name if target_channel else str(c)
    if target is not None:
        for c in range(target_c):
            curr_fig = input_c + c
            ax[curr_fig // num_cols, curr_fig % num_cols].imshow(target[c, :, :], cmap=CMAP)
            ax[curr_fig // num_cols, curr_fig % num_cols].set_title('Target ' + ax_title)

    if pred is not None:
        for c in range(pred_c):
            curr_fig = input_c + target_c + c
            ax[curr_fig // num_cols, curr_fig % num_cols].imshow(pred[c, :, :], cmap=CMAP)
            ax[curr_fig // num_cols, curr_fig % num_cols].set_title('Prediction ' + ax_title)
    else:
        curr_fig = input_c + target_c
        ax[curr_fig // num_cols, curr_fig % num_cols].axis('off')

    fig.suptitle(title)
    # plt.show()

    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, title + '.jpg'))

    plt.close(fig)
