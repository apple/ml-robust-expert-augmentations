#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2023 Apple Inc. All Rights Reserved.

import torch.utils.data as data_utils
import pickle


def load_data(data_path, device):
    with open(r"%s/train.pkl" % data_path, "rb") as output_file:
        t_train, x_train, true_param_train = pickle.load(output_file)
    t_train, x_train = t_train.to(device), x_train

    train = data_utils.TensorDataset(x_train)
    dl_train = data_utils.DataLoader(train, batch_size=100, shuffle=True)
    with open(r"%s/valid.pkl" % data_path, "rb") as output_file:
        t_valid, x_valid, true_param_valid = pickle.load(output_file)
    t_valid, x_valid = t_valid.to(device), x_valid.to(device)
    with open(r"%s/test_shifted.pkl" % data_path, "rb") as output_file:
        t_test_shifted, x_test_shifted, true_param_test_shifted = pickle.load(output_file)
    t_test_shifted, x_test_shifted = t_test_shifted.to(device), x_test_shifted.to(device)
    return (t_train, x_train, true_param_train), dl_train, (t_valid, x_valid, true_param_valid), (t_test_shifted, x_test_shifted, true_param_test_shifted)