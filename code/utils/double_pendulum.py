#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils


def xy_to_theta(dx, dy):
    theta = np.arctan2(dy, dx) + math.pi / 2
    # cond_1 = ((dy > 0) & (theta < 0))
    # cond_2 = ((dy < 0) & (theta > 0))
    # else_cond = 1 - (cond_1 & cond_2)
    # theta = (theta + math.pi) * cond_1 + cond_2 * (theta - math.pi) + else_cond * theta
    # theta = theta + math.pi/2
    # theta = theta * (theta < math.pi) + (theta - 2*math.pi) * (theta >= math.pi)
    return theta


def from_raw_pixels_to_angle(markers):
    cos_1 = markers[:, 3] - markers[:, 1]
    sin_1 = markers[:, 0] - markers[:, 2]
    theta_1 = xy_to_theta(cos_1, sin_1)

    cos_2 = markers[:, 5] - markers[:, 3]
    sin_2 = markers[:, 2] - markers[:, 4]
    theta_2 = xy_to_theta(cos_2, sin_2)

    return theta_1, theta_2


def video_to_frames(video_path, max_amount=np.Inf):
    '''Convert a video into its frames.'''
    frames = []
    # load the video
    vidcap = cv2.VideoCapture(video_path)
    while vidcap.isOpened():
        success, frame = vidcap.read()
        frames.append(frame)
        if len(frames) >= max_amount:
            break
    return frames


def get_dataloaders(path, files=['0'], sub_sampling_rate=1, seq_len=100, seq_gap=10, b_size=100):
    train, val, test = [], [], []
    for f in files:
        marker_positions = pd.read_csv(path + f +'.csv',
                                       header=None).values
        angles = from_raw_pixels_to_angle(marker_positions)
        angles_dataset = torch.cat((torch.tensor(angles[0]).unsqueeze(1), torch.tensor(angles[1]).unsqueeze(1)), 1)
        angles_dataset = angles_dataset[::sub_sampling_rate]
        length_tot = angles_dataset.shape[0]

        samples_ids = torch.arange(seq_len).unsqueeze(0) + torch.arange(0, length_tot-seq_len, seq_gap).unsqueeze(1)

        train_s = int(samples_ids.shape[0] * 0.6)
        val_s = int(samples_ids.shape[0] * .2)
        train.append(angles_dataset[samples_ids[:train_s]])
        val.append(angles_dataset[samples_ids[train_s:train_s+val_s]])
        test.append(angles_dataset[samples_ids[train_s+val_s:]])

    train = torch.cat(train, 0)
    val = torch.cat(val, 0)
    test = torch.cat(test, 0)

    train = data_utils.TensorDataset(train.float())
    dl_train = data_utils.DataLoader(train, batch_size=b_size, shuffle=True)
    val = data_utils.TensorDataset(val.float())
    dl_val = data_utils.DataLoader(val, batch_size=b_size, shuffle=False)
    test = data_utils.TensorDataset(test.float())
    dl_test = data_utils.DataLoader(test, batch_size=b_size, shuffle=False)

    frequency = int(400/sub_sampling_rate)
    time = torch.arange(0, seq_len / frequency, 1 / frequency)

    return dl_train, dl_val, dl_test, time


