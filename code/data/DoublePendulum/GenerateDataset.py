#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import math

import pandas as pd

import code.simulators.DoublePendulum as DP
import torch
from torchdiffeq import odeint
from tqdm import tqdm
import os
import pickle
import argparse

from code.utils.double_pendulum import from_raw_pixels_to_angle


def get_datatensors(files=['0'], sub_sampling_rate=1, seq_len=100, seq_gap=1):
    train, val, test = [], [], []
    for f in files:
        marker_positions = pd.read_csv(f'code/data/DoublePendulum/dpc_dataset_csv/{f}.csv',
                                       header=None).values
        angles = from_raw_pixels_to_angle(marker_positions)
        angles_dataset = torch.cat((torch.tensor(angles[0]).unsqueeze(1), torch.tensor(angles[1]).unsqueeze(1)), 1)
        angles_dataset = angles_dataset[::sub_sampling_rate]
        length_tot = angles_dataset.shape[0] // 2

        samples_ids = torch.arange(seq_len).unsqueeze(0) + torch.arange(0, length_tot - seq_len, seq_gap).unsqueeze(
            1) + angles_dataset.shape[0] // 2 - 1

        train_s = int(samples_ids.shape[0] * 0.4)
        val_s = int(samples_ids.shape[0] * .3)
        train.append(angles_dataset[samples_ids[-train_s:]])
        val.append(angles_dataset[samples_ids[-train_s - val_s:-train_s]])
        test.append(angles_dataset[samples_ids[:-train_s - val_s]])

    train = torch.cat(train, 0).float()
    train = train[torch.randperm(train.shape[0])]

    val = torch.cat(val, 0).float()
    val = val[torch.randperm(val.shape[0])]

    test = torch.cat(test, 0).float()
    test = test[torch.randperm(test.shape[0])]

    frequency = int(400 / sub_sampling_rate)
    time = torch.arange(0, seq_len / frequency, 1 / frequency)

    return train, val, test, time


if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--nb_train', type=int, default=1000)
    # Parse the argument
    args = parser.parse_args()

    nb_data_train = args.nb_train
    path = 'code/data/DoublePendulum'
    if not os.path.exists(path):
        os.makedirs(path)

    train, valid, test, time = get_datatensors(files=['%d' % i for i in range(21)], sub_sampling_rate=4, seq_len=21,
                                             seq_gap=1)

    with open(r"%s/train.pkl" % path, "wb") as output_file:
        dx, dy = train[:, 0, :], train[:, 2, :]
        diff = (dx - dy).unsqueeze(2)
        choices = torch.cat((diff - 2 * math.pi, diff, diff + 2 * math.pi), 2)
        _, choice = torch.min(choices ** 2, 2)  # np.arctan2(np.sin(x-y), np.cos(x-y)) * 200
        omegas = torch.gather(choices, dim=2, index=choice.unsqueeze(2)).squeeze(2) / (2 * time[1])
        init_states = torch.cat((train[:, 1, :], -omegas), 1)
        init_states = {"\\theta_0": init_states[:, 0], "\\theta_1": init_states[:, 1],
                       "\\dot \\theta_0": init_states[:, 2], "\\dot \\theta_1": init_states[:, 3]}

        pickle.dump([time[1:] - time[1], train[:, 1:].unsqueeze(3).unsqueeze(3), init_states], output_file)

    with open(r"%s/valid_shifted.pkl" % path, "wb") as output_file:
        dx, dy = valid[:, 0, :], valid[:, 2, :]
        diff = (dx - dy).unsqueeze(2)
        choices = torch.cat((diff - 2 * math.pi, diff, diff + 2 * math.pi), 2)
        _, choice = torch.min(choices ** 2, 2)  # np.arctan2(np.sin(x-y), np.cos(x-y)) * 200
        omegas = torch.gather(choices, dim=2, index=choice.unsqueeze(2)).squeeze(2) / (2 * time[1])
        init_states = torch.cat((valid[:, 1, :], -omegas), 1)
        init_states = {"\\theta_0": init_states[:, 0], "\\theta_1": init_states[:, 1],
                       "\\dot \\theta_0": init_states[:, 2], "\\dot \\theta_1": init_states[:, 3]}

        pickle.dump([time[1:] - time[1], valid[:, 1:].unsqueeze(3).unsqueeze(3), init_states], output_file)

    with open(r"%s/valid.pkl" % path, "wb") as output_file:
        dx, dy = valid[:, 0, :], valid[:, 2, :]
        diff = (dx - dy).unsqueeze(2)
        choices = torch.cat((diff - 2 * math.pi, diff, diff + 2 * math.pi), 2)
        _, choice = torch.min(choices ** 2, 2)  # np.arctan2(np.sin(x-y), np.cos(x-y)) * 200
        omegas = torch.gather(choices, dim=2, index=choice.unsqueeze(2)).squeeze(2) / (2 * time[1])
        init_states = torch.cat((valid[:, 1, :], -omegas), 1)
        init_states = {"\\theta_0": init_states[:, 0], "\\theta_1": init_states[:, 1],
                       "\\dot \\theta_0": init_states[:, 2], "\\dot \\theta_1": init_states[:, 3]}
        pickle.dump([time[1:] - time[1], valid[:, 1:].unsqueeze(3).unsqueeze(3), init_states], output_file)

    with open(r"%s/test.pkl" % path, "wb") as output_file:
        dx, dy = test[:, 0, :], test[:, 2, :]
        diff = (dx - dy).unsqueeze(2)
        choices = torch.cat((diff - 2 * math.pi, diff, diff + 2 * math.pi), 2)
        _, choice = torch.min(choices ** 2, 2)  # np.arctan2(np.sin(x-y), np.cos(x-y)) * 200
        omegas = torch.gather(choices, dim=2, index=choice.unsqueeze(2)).squeeze(2) / (2 * time[1])
        init_states = torch.cat((test[:, 1, :], -omegas), 1)
        init_states = {"\\theta_0": init_states[:, 0], "\\theta_1": init_states[:, 1],
                       "\\dot \\theta_0": init_states[:, 2], "\\dot \\theta_1": init_states[:, 3]}

        pickle.dump([time[1:] - time[1], test[:, 1:].unsqueeze(3).unsqueeze(3), init_states], output_file)

    with open(r"%s/test_shifted.pkl" % path, "wb") as output_file:
        dx, dy = test[:, 0, :], test[:, 2, :]
        diff = (dx - dy).unsqueeze(2)
        choices = torch.cat((diff - 2 * math.pi, diff, diff + 2 * math.pi), 2)
        _, choice = torch.min(choices ** 2, 2)  # np.arctan2(np.sin(x-y), np.cos(x-y)) * 200
        omegas = torch.gather(choices, dim=2, index=choice.unsqueeze(2)).squeeze(2) / (2 * time[1])
        init_states = torch.cat((test[:, 1, :], -omegas), 1)
        init_states = {"\\theta_0": init_states[:, 0], "\\theta_1": init_states[:, 1],
                       "\\dot \\theta_0": init_states[:, 2], "\\dot \\theta_1": init_states[:, 3]}

        pickle.dump([time[1:] - time[1], test[:, 1:].unsqueeze(3).unsqueeze(3), init_states], output_file)
